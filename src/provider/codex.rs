//! OpenAI Codex Responses provider (WebSocket + SSE fallback).

use std::sync::Arc;

use futures::stream;
use serde_json::Value;
use tokio_tungstenite::tungstenite;
use crate::env::resolve_api_key;
use crate::events::Event;
use crate::provider::responses;
use crate::types::*;

/// Start a Codex stream (WebSocket with SSE fallback).
pub fn stream_codex<'a>(
    model: &'a Model,
    context: &'a Context,
    opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    let api_key = resolve_api_key(model, opts);
    if api_key.is_none() {
        let err = Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                format!("no API key for provider: {}", model.provider),
            )),
            message: None,
        };
        return Box::pin(stream::once(async { err }));
    }
    let api_key = api_key.unwrap();

    Box::pin(async_stream::stream! {
        // Try WebSocket first, fall back to SSE
        let ws_url = format!(
            "{}/responses?model={}&stream=true",
            model.base_url.trim_end_matches('/').replace("https://", "wss://").replace("http://", "ws://"),
            &model.id
        );

        let ws_result = try_websocket(&ws_url, &api_key, model, context, opts).await;
        match ws_result {
            Ok(events) => {
                for evt in events {
                    yield evt;
                }
            }
            Err(_ws_err) => {
                // Fallback to the richer Responses SSE implementation so Codex keeps
                // parity with current Responses event/payload handling.
                let mut fallback = responses::stream_responses(model, context, opts);
                use futures::StreamExt;
                while let Some(evt) = fallback.next().await {
                    yield evt;
                }
            }
        }
    })
}

async fn try_websocket(
    ws_url: &str,
    api_key: &str,
    model: &Model,
    context: &Context,
    opts: &StreamOptions,
) -> Result<Vec<Event>, String> {
    use tokio_tungstenite::connect_async;
    use futures::SinkExt;

    let request = tungstenite::http::Request::builder()
        .uri(ws_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Sec-WebSocket-Protocol", "openai-beta.responses")
        .body(())
        .map_err(|e| e.to_string())?;

    let (mut ws, _) = connect_async(request)
        .await
        .map_err(|e| e.to_string())?;

    // Send the request payload
    let payload = build_codex_payload(model, context, opts);
    ws.send(tungstenite::Message::Text(serde_json::to_string(&payload).unwrap().into()))
        .await
        .map_err(|e| e.to_string())?;

    use futures::StreamExt;
    let mut state = CodexWsState::new(model);

    while let Some(msg) = ws.next().await {
        let msg = msg.map_err(|e| e.to_string())?;
        let text = match msg {
            tungstenite::Message::Text(t) => t.to_string(),
            tungstenite::Message::Close(_) => break,
            _ => continue,
        };

        let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let is_done = state.process_event(&data);
        if is_done {
            break;
        }
    }

    Ok(state.finish())
}

#[derive(Debug, Clone)]
struct CodexWsState {
    partial: Message,
    events: Vec<Event>,
    current_text: String,
    text_started: bool,
    current_thinking: String,
    current_tool_call_id: Option<String>,
    current_tool_item_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_args: String,
}

impl CodexWsState {
    fn new(model: &Model) -> Self {
        let partial = Message {
            role: Role::Assistant,
            content: Vec::new(),
            timestamp: 0,
            api: Some(model.api.clone()),
            provider: Some(model.provider.clone()),
            model: Some(model.id.clone()),
            response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None,
            stop_reason: None,
            error_message: None,
            tool_call_id: None,
            tool_name: None,
            is_error: false,
            details: None,
        };
        let events = vec![Event::Start { partial: partial.clone() }];
        Self {
            partial,
            events,
            current_text: String::new(),
            text_started: false,
            current_thinking: String::new(),
            current_tool_call_id: None,
            current_tool_item_id: None,
            current_tool_name: None,
            current_tool_args: String::new(),
        }
    }

    fn process_event(&mut self, data: &Value) -> bool {
        let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match event_type {
            "response.created" => {
                if let Some(response) = data.get("response") {
                    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                        self.partial.response_id = Some(id.to_string());
                    }
                    if let Some(model_name) = response.get("model").and_then(|v| v.as_str()) {
                        self.partial.response_model = Some(model_name.to_string());
                    }
                }
            }
            "response.output_item.added" => {
                if let Some(item) = data.get("item") {
                    match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                        "function_call" => {
                            self.current_tool_call_id = item.get("call_id").and_then(|v| v.as_str()).map(|s| s.to_string());
                            self.current_tool_item_id = item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                            self.current_tool_name = item.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
                            self.current_tool_args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            if let (Some(id), Some(name)) = (self.current_tool_call_id.clone(), self.current_tool_name.clone()) {
                                self.events.push(Event::ToolCallStart { id, name });
                            }
                            if !self.current_tool_args.is_empty() {
                                self.events.push(Event::ToolCallDelta { delta: self.current_tool_args.clone() });
                            }
                        }
                        "reasoning" => self.events.push(Event::ThinkingStart),
                        "message" => {
                            if !self.text_started {
                                self.text_started = true;
                                self.events.push(Event::TextStart);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "response.output_text.delta" => {
                if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                    if !self.text_started {
                        self.text_started = true;
                        self.events.push(Event::TextStart);
                    }
                    self.current_text.push_str(delta);
                    self.events.push(Event::TextDelta { delta: delta.to_string() });
                }
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                    self.current_thinking.push_str(delta);
                    self.events.push(Event::ThinkingDelta { delta: delta.to_string() });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                    self.current_tool_args.push_str(delta);
                    self.events.push(Event::ToolCallDelta { delta: delta.to_string() });
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(arguments) = data.get("arguments").and_then(|v| v.as_str()) {
                    if arguments.starts_with(&self.current_tool_args) {
                        let extra = &arguments[self.current_tool_args.len()..];
                        if !extra.is_empty() {
                            self.current_tool_args.push_str(extra);
                            self.events.push(Event::ToolCallDelta { delta: extra.to_string() });
                        }
                    } else {
                        self.current_tool_args = arguments.to_string();
                    }
                }
            }
            "response.output_item.done" => {
                if let Some(item) = data.get("item") {
                    match item.get("type").and_then(|v| v.as_str()) {
                        Some("function_call") => {
                            let id = item.get("call_id").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| self.current_tool_call_id.clone()).unwrap_or_default();
                            let name = item.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| self.current_tool_name.clone()).unwrap_or_default();
                            let final_args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or(&self.current_tool_args);
                            let parsed: Value = crate::jsonparse::parse_streaming_json(final_args);
                            let parsed_map = match &parsed {
                                Value::Object(map) => map.clone().into_iter().collect(),
                                _ => std::collections::HashMap::new(),
                            };
                            self.partial.content.push(ContentBlock::ToolCall {
                                id: match item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| self.current_tool_item_id.clone()) {
                                    Some(item_id) if !id.is_empty() => format!("{}|{}", id, item_id),
                                    _ => id.clone(),
                                },
                                name: name.clone(),
                                arguments: parsed_map,
                                thought_signature: None,
                            });
                            self.events.push(Event::ToolCallEnd { id, name, arguments: parsed });
                            self.current_tool_call_id = None;
                            self.current_tool_item_id = None;
                            self.current_tool_name = None;
                            self.current_tool_args.clear();
                        }
                        Some("reasoning") => {
                            let thinking_text = item.get("summary").and_then(|v| v.as_array())
                                .map(|parts| parts.iter().filter_map(|p| p.get("text").and_then(|v| v.as_str())).collect::<Vec<_>>().join("\n\n"))
                                .filter(|s| !s.is_empty())
                                .or_else(|| item.get("content").and_then(|v| v.as_array()).map(|parts| parts.iter().filter_map(|p| p.get("text").and_then(|v| v.as_str())).collect::<Vec<_>>().join("\n\n")).filter(|s| !s.is_empty()))
                                .unwrap_or_else(|| self.current_thinking.clone());
                            self.partial.content.push(ContentBlock::Thinking {
                                thinking: thinking_text,
                                thinking_signature: Some(item.to_string()),
                                redacted: false,
                            });
                            self.events.push(Event::ThinkingEnd);
                            self.current_thinking.clear();
                        }
                        _ => {}
                    }
                }
            }
            "response.completed" => {
                if self.text_started {
                    self.events.push(Event::TextEnd);
                    self.text_started = false;
                }
                if let Some(response) = data.get("response") {
                    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                        self.partial.response_id = Some(id.to_string());
                    }
                    if let Some(model_name) = response.get("model").and_then(|v| v.as_str()) {
                        self.partial.response_model = Some(model_name.to_string());
                    }
                    if let Some(usage) = response.get("usage") {
                        self.partial.usage = Some(Usage {
                            input: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            output: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            ..Default::default()
                        });
                    }
                }
                self.partial.stop_reason = Some(if self.partial.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. })) { StopReason::ToolUse } else { StopReason::Stop });
                if !self.current_text.is_empty() && !self.partial.content.iter().any(|b| matches!(b, ContentBlock::Text { .. })) {
                    self.partial.content.push(ContentBlock::Text { text: self.current_text.clone(), text_signature: None });
                }
                return true;
            }
            _ => {}
        }
        false
    }

    fn finish(mut self) -> Vec<Event> {
        let reason = self.partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        self.events.push(Event::Done { reason, message: self.partial.clone() });
        self.events
    }
}

pub(crate) fn build_codex_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    responses::build_responses_payload(model, context, opts)
}

#[cfg(test)]
pub(crate) fn replay_codex_ws_events(model: &Model, events: &[Value]) -> Vec<Event> {
    let mut state = CodexWsState::new(model);
    for event in events {
        if state.process_event(event) {
            break;
        }
    }
    state.finish()
}
