//! OpenAI Chat Completions provider (also serves compatible APIs).

use std::collections::HashMap;
use std::sync::Arc;

use futures::stream::{self, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::compat::detect_compat;
use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start an OpenAI-compatible chat completions stream.
pub fn stream_openai<'a>(
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
    let compat = detect_compat(model);

    // Build request payload
    let payload = build_payload(model, context, opts, &compat);

    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("Accept", HeaderValue::from_static("text/event-stream"));

    if model.provider == "cloudflare-ai-gateway" {
        headers.insert("cf-aig-authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    } else {
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    }

    // Add model-level headers
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
        let resp = client
            .post(&url)
            .headers(headers)
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

        let mut text_started = false;
        let mut current_text = String::new();

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
            if evt.data == "[DONE]" {
                break;
            }
            let chunk: Value = match serde_json::from_str(&evt.data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(id) = chunk.get("id").and_then(|v| v.as_str()) {
                partial.response_id = Some(id.to_string());
            }

            if let Some(choices) = chunk.get("choices").and_then(|v| v.as_array()) {
                for choice in choices {
                    let delta = match choice.get("delta") {
                        Some(d) => d,
                        None => continue,
                    };

                    // Text content
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        if !content.is_empty() {
                            if !text_started {
                                text_started = true;
                                yield Event::TextStart;
                            }
                            current_text.push_str(content);
                            yield Event::TextDelta { delta: content.to_string() };
                        }
                    }

                    // Finish reason
                    if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        if text_started {
                            yield Event::TextEnd;
                        }
                        let stop = match reason {
                            "stop" => StopReason::Stop,
                            "length" => StopReason::Length,
                            "tool_calls" => StopReason::ToolUse,
                            _ => StopReason::Stop,
                        };
                        partial.stop_reason = Some(stop.clone());
                        if !current_text.is_empty() {
                            partial.content.push(ContentBlock::Text {
                                text: current_text.clone(),
                                text_signature: None,
                            });
                        }
                    }
                }
            }

            // Usage
            if let Some(usage) = chunk.get("usage") {
                partial.usage = Some(Usage {
                    input: usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                    output: usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                    total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                    ..Default::default()
                });
            }
        }

        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

fn build_payload(
    model: &Model,
    context: &Context,
    opts: &StreamOptions,
    compat: &crate::compat::OpenAICompletionsCompat,
) -> Value {
    let mut messages = Vec::new();

    // System prompt
    if let Some(ref prompt) = context.system_prompt {
        let role = if compat.supports_developer_role == Some(true) {
            "developer"
        } else {
            "system"
        };
        messages.push(json!({ "role": role, "content": prompt }));
    }

    // Conversation messages
    for msg in &context.messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::ToolResult => "tool",
        };
        let content: Value = if msg.content.len() == 1 {
            match &msg.content[0] {
                ContentBlock::Text { text, .. } => json!(text),
                _ => json!(format_content_blocks(&msg.content)),
            }
        } else {
            json!(format_content_blocks(&msg.content))
        };

        let mut m = json!({ "role": role_str, "content": content });
        if msg.role == Role::ToolResult {
            if let Some(ref id) = msg.tool_call_id {
                m["tool_call_id"] = json!(id);
            }
        }
        messages.push(m);
    }

    let max_tokens_field = compat.max_tokens_field.as_deref().unwrap_or("max_completion_tokens");

    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });

    if let Some(max) = opts.max_tokens {
        payload[max_tokens_field] = json!(max);
    } else if model.max_tokens > 0 {
        payload[max_tokens_field] = json!(model.max_tokens);
    }

    if let Some(temp) = opts.temperature {
        if compat.supports_temperature != Some(false) {
            payload["temperature"] = json!(temp);
        }
    }

    // Tools
    if !context.tools.is_empty() {
        let tools: Vec<Value> = context.tools.iter().map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        }).collect();
        payload["tools"] = json!(tools);
    }

    payload
}

fn format_content_blocks(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks.iter().map(|b| match b {
        ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
        ContentBlock::Image { data, mime_type } => json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{};base64,{}", mime_type, data)}
        }),
        ContentBlock::Thinking { thinking, .. } => json!({"type": "text", "text": thinking}),
        ContentBlock::ToolCall { id, name, arguments, .. } => json!({
            "type": "text",
            "text": format!("[tool_call: {} {}]", name, serde_json::to_string(arguments).unwrap_or_default())
        }),
    }).collect()
}
