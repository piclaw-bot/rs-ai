//! OpenAI Responses API provider (also serves Azure OpenAI Responses).

use std::sync::Arc;

use futures::stream;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start an OpenAI Responses stream.
pub fn stream_responses<'a>(
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

    let payload = build_responses_payload(model, context, opts);
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());

    if let Some(ref model_headers) = model.headers {
        for (k, v) in model_headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                HeaderValue::from_str(v),
            ) {
                headers.insert(name, val);
            }
        }
    }

    Box::pin(async_stream::stream! {
        let client = reqwest::Client::new();
        let resp = client.post(&url).headers(headers).json(&payload).send().await;

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
                error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                    format!("HTTP {}: {}", status, body),
                )),
                message: None,
            };
            return;
        }

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

        yield Event::Start { partial: partial.clone() };

        let bytes = resp.bytes().await.unwrap_or_default();
        let events = sse::parse(bytes.as_ref());

        let mut current_text = String::new();
        let mut text_started = false;

        for evt in events {
            if evt.event == sse::EVENT_ERROR {
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                        format!("SSE error: {}", evt.data),
                    )),
                    message: Some(partial.clone()),
                };
                return;
            }

            let data: Value = match serde_json::from_str(&evt.data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match event_type {
                "response.created" => {
                    if let Some(id) = data.get("response").and_then(|r| r.get("id")).and_then(|v| v.as_str()) {
                        partial.response_id = Some(id.to_string());
                    }
                }
                "response.output_item.added" => {
                    // item started
                }
                "response.content_part.added" => {
                    if !text_started {
                        text_started = true;
                        yield Event::TextStart;
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                        current_text.push_str(delta);
                        yield Event::TextDelta { delta: delta.to_string() };
                    }
                }
                "response.content_part.done" => {
                    if text_started {
                        text_started = false;
                        yield Event::TextEnd;
                    }
                }
                "response.completed" => {
                    if let Some(response) = data.get("response") {
                        if let Some(usage) = response.get("usage") {
                            partial.usage = Some(Usage {
                                input: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                output: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                ..Default::default()
                            });
                        }
                        partial.stop_reason = Some(StopReason::Stop);
                    }
                }
                _ => {}
            }
        }

        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text {
                text: current_text,
                text_signature: None,
            });
        }
        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

fn build_responses_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
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
        let content: Value = if msg.content.len() == 1 {
            match &msg.content[0] {
                ContentBlock::Text { text, .. } => json!(text),
                _ => json!(""),
            }
        } else {
            json!(msg.content.iter().map(|b| match b {
                ContentBlock::Text { text, .. } => json!({"type": "input_text", "text": text}),
                ContentBlock::Image { data, mime_type } => json!({
                    "type": "input_image", "image_url": format!("data:{};base64,{}", mime_type, data)
                }),
                _ => json!({"type": "input_text", "text": ""}),
            }).collect::<Vec<_>>())
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
    if let Some(temp) = opts.temperature {
        payload["temperature"] = json!(temp);
    }

    if !context.tools.is_empty() {
        let tools: Vec<Value> = context.tools.iter().map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        }).collect();
        payload["tools"] = json!(tools);
    }

    payload
}
