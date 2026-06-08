//! OpenAI Codex Responses provider (WebSocket + SSE fallback).

use std::sync::Arc;

use futures::stream;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite;

use crate::diagnostics::transport_failure_diagnostic;
use crate::env::resolve_api_key;
use crate::events::Event;
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
                // Fallback to SSE
                let sse_events = sse_fallback(model, context, opts, &api_key).await;
                for evt in sse_events {
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

    // Read events
    use futures::StreamExt;
    let mut events = Vec::new();
    let mut partial = Message {
        role: Role::Assistant,
        content: Vec::new(),
        timestamp: 0,
        api: Some(model.api.clone()),
        provider: Some(model.provider.clone()),
        model: Some(model.id.clone()),
        response_id: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        tool_call_id: None,
        tool_name: None,
        is_error: false,
    };

    events.push(Event::Start { partial: partial.clone() });

    let mut current_text = String::new();
    let mut text_started = false;

    while let Some(msg) = ws.next().await {
        let msg = msg.map_err(|e| e.to_string())?;
        let text = match msg {
            tungstenite::Message::Text(t) => t.to_string(),
            tungstenite::Message::Close(_) => break,
            _ => continue,
        };

        let data: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                    if !text_started {
                        text_started = true;
                        events.push(Event::TextStart);
                    }
                    current_text.push_str(delta);
                    events.push(Event::TextDelta { delta: delta.to_string() });
                }
            }
            "response.completed" => {
                if text_started {
                    events.push(Event::TextEnd);
                }
                if let Some(response) = data.get("response") {
                    if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                        partial.response_id = Some(id.to_string());
                    }
                    if let Some(usage) = response.get("usage") {
                        partial.usage = Some(Usage {
                            input: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            output: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            ..Default::default()
                        });
                    }
                }
                partial.stop_reason = Some(StopReason::Stop);
                if !current_text.is_empty() {
                    partial.content.push(ContentBlock::Text { text: current_text.clone(), text_signature: None });
                }
                events.push(Event::Done { reason: StopReason::Stop, message: partial.clone() });
                break;
            }
            _ => {}
        }
    }

    Ok(events)
}

async fn sse_fallback(
    model: &Model,
    context: &Context,
    opts: &StreamOptions,
    api_key: &str,
) -> Vec<Event> {
    let payload = build_codex_payload(model, context, opts);
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());

    let client = reqwest::Client::new();
    let resp = match client.post(&url).headers(headers).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return vec![Event::Error {
                reason: StopReason::Error,
                error: Arc::from(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                message: None,
            }];
        }
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return vec![Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                format!("HTTP {}: {}", status, body),
            )),
            message: None,
        }];
    }

    let bytes = resp.bytes().await.unwrap_or_default();
    let sse_events = sse::parse(bytes.as_ref());

    let mut events = Vec::new();
    let mut partial = Message {
        role: Role::Assistant,
        content: Vec::new(),
        timestamp: 0,
        api: Some(model.api.clone()),
        provider: Some(model.provider.clone()),
        model: Some(model.id.clone()),
        response_id: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        tool_call_id: None,
        tool_name: None,
        is_error: false,
    };

    // Attach transport failure diagnostic
    let _diag = transport_failure_diagnostic("WebSocket setup failed; fell back to SSE");

    events.push(Event::Start { partial: partial.clone() });
    let mut current_text = String::new();
    let mut text_started = false;

    for evt in sse_events {
        if evt.event == sse::EVENT_ERROR {
            events.push(Event::Error {
                reason: StopReason::Error,
                error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(evt.data)),
                message: Some(partial.clone()),
            });
            return events;
        }

        let data: Value = match serde_json::from_str(&evt.data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                    if !text_started {
                        text_started = true;
                        events.push(Event::TextStart);
                    }
                    current_text.push_str(delta);
                    events.push(Event::TextDelta { delta: delta.to_string() });
                }
            }
            "response.completed" => {
                if text_started {
                    events.push(Event::TextEnd);
                }
                partial.stop_reason = Some(StopReason::Stop);
                if !current_text.is_empty() {
                    partial.content.push(ContentBlock::Text { text: current_text.clone(), text_signature: None });
                }
                events.push(Event::Done { reason: StopReason::Stop, message: partial.clone() });
                break;
            }
            _ => {}
        }
    }

    events
}

fn build_codex_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let mut input = Vec::new();

    if let Some(ref prompt) = context.system_prompt {
        input.push(json!({"role": "system", "content": prompt}));
    }

    for msg in &context.messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::ToolResult => "user",
        };
        let content = match msg.content.first() {
            Some(ContentBlock::Text { text, .. }) => json!(text),
            _ => json!(""),
        };
        input.push(json!({"role": role_str, "content": content}));
    }

    let mut payload = json!({
        "model": model.id,
        "input": input,
        "stream": true,
    });

    if let Some(max) = opts.max_tokens {
        payload["max_output_tokens"] = json!(max);
    } else if model.max_tokens > 0 {
        payload["max_output_tokens"] = json!(model.max_tokens);
    }

    payload
}
