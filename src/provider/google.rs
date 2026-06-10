//! Google Generative AI (Gemini) provider.

use std::sync::Arc;

use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start a Google Generative AI stream.
pub fn stream_google<'a>(
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

    let payload = build_google_payload(model, context, opts);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse&key={}",
        model.base_url.trim_end_matches('/'),
        url_encode(&model.id),
        url_encode(&api_key),
    );

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

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

        yield Event::Start { partial: partial.clone() };

        let mut parser = sse::SseParser::default();
        let mut byte_stream = resp.bytes_stream();

        let mut current_text = String::new();
        let mut text_started = false;
        let mut thinking_started = false;
        let mut current_thinking = String::new();
        let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk_bytes = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Event::Error {
                        reason: StopReason::Error,
                        error: Arc::from(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        message: Some(partial.clone()),
                    };
                    return;
                }
            };

            let chunk_text = String::from_utf8_lossy(&chunk_bytes);
            for evt in parser.feed(&chunk_text) {
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

                let chunk: Value = match serde_json::from_str(&evt.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if let Some(candidates) = chunk.get("candidates").and_then(|v| v.as_array()) {
                    for candidate in candidates {
                        if let Some(parts) = candidate.pointer("/content/parts").and_then(|v| v.as_array()) {
                            for part in parts {
                                let is_thought = part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                    if !text.is_empty() {
                                        if is_thought {
                                            if !thinking_started {
                                                thinking_started = true;
                                                yield Event::ThinkingStart;
                                            }
                                            current_thinking.push_str(text);
                                            yield Event::ThinkingDelta { delta: text.to_string() };
                                        } else {
                                            if !text_started {
                                                text_started = true;
                                                yield Event::TextStart;
                                            }
                                            current_text.push_str(text);
                                            yield Event::TextDelta { delta: text.to_string() };
                                        }
                                    }
                                }
                                if let Some(fc) = part.get("functionCall") {
                                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
                                    let id = format!("call_{}", tool_calls.len());
                                    yield Event::ToolCallStart { id: id.clone(), name: name.clone() };
                                    yield Event::ToolCallDelta { delta: serde_json::to_string(&args).unwrap_or_default() };
                                    yield Event::ToolCallEnd { id: id.clone(), name: name.clone(), arguments: args.clone() };
                                    tool_calls.push((id, name, args));
                                }
                            }
                        }
                        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str()) {
                            if text_started {
                                yield Event::TextEnd;
                                text_started = false;
                            }
                            if thinking_started {
                                yield Event::ThinkingEnd;
                                thinking_started = false;
                            }
                            partial.stop_reason = Some(match reason {
                                "STOP" if !tool_calls.is_empty() => StopReason::ToolUse,
                                "STOP" => StopReason::Stop,
                                "MAX_TOKENS" => StopReason::Length,
                                _ => StopReason::Stop,
                            });
                        }
                    }
                }

                if let Some(usage) = chunk.get("usageMetadata") {
                    partial.usage = Some(Usage {
                        input: usage.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        output: usage.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        cache_read: usage.get("cachedContentTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        total_tokens: usage.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        ..Default::default()
                    });
                }
            }
        }

        if let Some(evt) = parser.finish() {
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
        }

        if !current_thinking.is_empty() {
            partial.content.push(ContentBlock::Thinking {
                thinking: current_thinking,
                thinking_signature: None,
                redacted: false,
            });
        }
        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text {
                text: current_text,
                text_signature: None,
            });
        }
        for (id, name, args) in tool_calls {
            let arguments = match args {
                serde_json::Value::Object(map) => map.into_iter().collect(),
                _ => std::collections::HashMap::new(),
            };
            partial.content.push(ContentBlock::ToolCall {
                id,
                name,
                arguments,
                thought_signature: None,
            });
        }
        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

/// Build Google Generative AI request payload (public for Gemini CLI reuse).
pub fn build_google_payload_public(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    build_google_payload(model, context, opts)
}

fn build_google_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let mut contents = Vec::new();

    for msg in &context.messages {
        let role = match msg.role {
            Role::User | Role::ToolResult => "user",
            Role::Assistant => "model",
        };
        let parts: Vec<Value> = msg.content.iter().map(|b| match b {
            ContentBlock::Text { text, .. } => json!({"text": text}),
            ContentBlock::Image { data, mime_type } => json!({
                "inlineData": {"mimeType": mime_type, "data": data}
            }),
            ContentBlock::Thinking { thinking, .. } => json!({"text": thinking}),
            ContentBlock::ToolCall { name, arguments, .. } => json!({
                "functionCall": {"name": name, "args": arguments}
            }),
        }).collect();
        contents.push(json!({"role": role, "parts": parts}));
    }

    let mut payload = json!({"contents": contents});

    if let Some(ref prompt) = context.system_prompt {
        payload["systemInstruction"] = json!({"parts": [{"text": prompt}]});
    }

    let mut config = json!({});
    if let Some(max) = opts.max_tokens {
        config["maxOutputTokens"] = json!(max);
    } else if model.max_tokens > 0 {
        config["maxOutputTokens"] = json!(model.max_tokens);
    }
    if let Some(temp) = opts.temperature {
        config["temperature"] = json!(temp);
    }
    if config != json!({}) {
        payload["generationConfig"] = config;
    }

    if !context.tools.is_empty() {
        let decls: Vec<Value> = context.tools.iter().map(|t| {
            json!({"name": t.name, "description": t.description, "parameters": t.parameters})
        }).collect();
        payload["tools"] = json!([{"functionDeclarations": decls}]);
    }

    payload
}

fn url_encode(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('+', "%2B")
        .replace('/', "%2F")
}
