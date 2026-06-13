//! OpenAI Codex Responses provider (WebSocket + SSE fallback).

use std::sync::Arc;

use futures::stream;
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite;
use crate::env::resolve_api_key;
use crate::events::Event;
use crate::provider::responses;
use crate::transports::sse;
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
                // Fallback to SSE using the Codex request body and headers.
                let payload = build_codex_payload(model, context, opts);
                let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
                let client = reqwest::Client::new();
                let resp = client
                    .post(&url)
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .header("authorization", format!("Bearer {}", api_key))
                    .header("originator", "pi")
                    .json(&payload)
                    .send()
                    .await;
                let resp = match resp {
                    Ok(r) => r,
                    Err(e) => {
                        yield Event::Error {
                            reason: StopReason::Error,
                            error: Arc::from(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                            message: None,
                        };
                        return;
                    }
                };
                if !resp.status().is_success() {
                    let status = resp.status().as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    yield Event::Error {
                        reason: StopReason::Error,
                        error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(format!("HTTP {}: {}", status, body))),
                        message: None,
                    };
                    return;
                }

                use futures::StreamExt;
                let mut parser = sse::SseParser::default();
                let mut state = CodexWsState::new(model);
                let mut byte_stream = resp.bytes_stream();
                let mut emitted = 0usize;
                let mut done = false;
                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = match chunk_result {
                        Ok(c) => c,
                        Err(e) => {
                            yield Event::Error {
                                reason: StopReason::Error,
                                error: Arc::from(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                                message: None,
                            };
                            return;
                        }
                    };
                    for evt in parser.feed_bytes(&chunk) {
                        if evt.event == sse::EVENT_ERROR { continue; }
                        if let Ok(data) = serde_json::from_str::<Value>(&evt.data) {
                            done = state.process_event(&data);
                            if done { break; }
                        }
                    }
                    while emitted < state.events.len() {
                        yield state.events[emitted].clone();
                        emitted += 1;
                    }
                    if done { break; }
                }
                let final_events = state.finish();
                while emitted < final_events.len() {
                    yield final_events[emitted].clone();
                    emitted += 1;
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
        .header("originator", "pi")
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
    model_cost: crate::types::ModelCost,
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
            timestamp: crate::utils::now_millis(),
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
            model_cost: model.cost.clone(),
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
                        "message"
                            if !self.text_started => {
                                self.text_started = true;
                                self.events.push(Event::TextStart);
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
                        let cached = usage.pointer("/input_tokens_details/cached_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let input_total = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let input = input_total.saturating_sub(cached);
                        let output = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let total = usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or((input + output + cached) as u64) as u32;
                        let mut u = Usage {
                            input, output, cache_read: cached, cache_write: 0, total_tokens: total, cost: Default::default(),
                        };
                        let m = 1_000_000.0;
                        u.cost = crate::types::CostBreakdown {
                            input: f64::from(u.input) * self.model_cost.input / m,
                            output: f64::from(u.output) * self.model_cost.output / m,
                            cache_read: f64::from(u.cache_read) * self.model_cost.cache_read / m,
                            cache_write: f64::from(u.cache_write) * self.model_cost.cache_write / m,
                            total: 0.0,
                        };
                        u.cost.total = u.cost.input + u.cost.output + u.cost.cache_read + u.cost.cache_write;
                        self.partial.usage = Some(u);
                    }
                }
                self.partial.stop_reason = Some(if self.partial.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. })) { StopReason::ToolUse } else { StopReason::Stop });
                if !self.current_text.is_empty() && !self.partial.content.iter().any(|b| matches!(b, ContentBlock::Text { .. })) {
                    self.partial.content.push(ContentBlock::Text { text: self.current_text.clone(), text_signature: None });
                }
                return true;
            }
            "error" | "response.failed" => {
                let msg = data.pointer("/message").and_then(|v| v.as_str())
                    .or_else(|| data.pointer("/error/message").and_then(|v| v.as_str()))
                    .or_else(|| data.pointer("/response/error/message").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "Codex stream error".to_string());
                let code = data.get("code").and_then(|v| v.as_str()).map(|c| format!("Error Code {}: ", c)).unwrap_or_default();
                let full = format!("{}{}", code, msg);
                self.partial.stop_reason = Some(StopReason::Error);
                self.partial.error_message = Some(full.clone());
                self.events.push(Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(full)),
                    message: Some(self.partial.clone()),
                });
                return true;
            }
            _ => {}
        }
        false
    }

    fn finish(mut self) -> Vec<Event> {
        // If an error event was already emitted, do not also emit Done.
        if matches!(self.partial.stop_reason, Some(StopReason::Error)) {
            return self.events;
        }
        let reason = self.partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        self.events.push(Event::Done { reason, message: self.partial.clone() });
        self.events
    }
}

pub(crate) fn build_codex_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    // Reuse the Responses input/tool conversion, then restructure for Codex:
    // the system prompt moves to `instructions` and is removed from `input`.
    let base = responses::build_responses_payload(model, context, opts);
    let mut input = base.get("input").cloned().unwrap_or_else(|| json!([]));
    if let Some(arr) = input.as_array_mut() {
        arr.retain(|m| {
            !matches!(m.get("role").and_then(|r| r.as_str()), Some("system") | Some("developer"))
        });
    }

    let instructions = context.system_prompt.clone().unwrap_or_else(|| "You are a helpful assistant.".to_string());
    let mut body = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": instructions,
        "input": input,
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(ref session_id) = opts.session_id {
        body["prompt_cache_key"] = json!(crate::prompt_cache::clamp_openai_prompt_cache_key(session_id));
    }
    if let Some(temp) = opts.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(ref service_tier) = opts.service_tier {
        body["service_tier"] = json!(service_tier);
    }
    if !context.tools.is_empty()
        && let Some(tools) = base.get("tools") {
            // Codex uses strict: null (not false) on tool definitions.
            let mut tools = tools.clone();
            if let Some(arr) = tools.as_array_mut() {
                for t in arr.iter_mut() {
                    t["strict"] = Value::Null;
                }
            }
            body["tools"] = tools;
    }
    if let Some(level) = opts.reasoning.as_ref().and_then(|l| crate::simple_options::clamp_reasoning_for_model(model, l)) {
        let key = format!("{:?}", level).to_lowercase();
        let effort = model.thinking_level_map.as_ref()
            .and_then(|m| m.get(&key))
            .and_then(|v| v.clone())
            .unwrap_or(key);
        body["reasoning"] = json!({
            "effort": effort,
            "summary": opts.reasoning_summary.clone().unwrap_or_else(|| "auto".to_string()),
        });
    }
    body
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
