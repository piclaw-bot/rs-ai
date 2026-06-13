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
    headers.insert("accept", HeaderValue::from_static("text/event-stream"));

    Box::pin(async_stream::stream! {
        let client = reqwest::Client::new();
        let request = client.post(&url).headers(headers).json(&payload);
        let retry_cfg = crate::retry::retry_config_from_options(opts);
        let resp = crate::retry::do_with_retry(&client, request, &retry_cfg).await;

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

        yield Event::Start { partial: partial.clone() };

        let mut parser = sse::SseParser::default();
        let mut byte_stream = resp.bytes_stream();

        let mut current_text = String::new();
        let mut text_started = false;
        let mut thinking_started = false;
        let mut current_thinking = String::new();
        let mut current_thinking_signature: Option<String> = None;
        let mut tool_calls: Vec<(String, String, serde_json::Value, Option<String>)> = Vec::new();

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

            for evt in parser.feed_bytes(&chunk_bytes) {
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

                if let Some(err) = chunk.get("error") {
                    let msg = err.get("message").and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| err.to_string());
                    partial.stop_reason = Some(StopReason::Error);
                    partial.error_message = Some(msg.clone());
                    yield Event::Error {
                        reason: StopReason::Error,
                        error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                        message: Some(partial.clone()),
                    };
                    return;
                }

                if partial.response_id.is_none()
                    && let Some(rid) = chunk.get("responseId").and_then(|v| v.as_str()) {
                    partial.response_id = Some(rid.to_string());
                }
                if let Some(candidates) = chunk.get("candidates").and_then(|v| v.as_array()) {
                    for candidate in candidates {
                        if let Some(parts) = candidate.pointer("/content/parts").and_then(|v| v.as_array()) {
                            for part in parts {
                                let is_thought = part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                if is_thought
                                    && let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                                    current_thinking_signature = Some(sig.to_string());
                                }
                                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                                    && !text.is_empty() {
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
                                if let Some(fc) = part.get("functionCall") {
                                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
                                    let id = format!("call_{}", tool_calls.len());
                                    let sig = part.get("thoughtSignature").and_then(|v| v.as_str()).map(|s| s.to_string());
                                    yield Event::ToolCallStart { id: id.clone(), name: name.clone() };
                                    yield Event::ToolCallDelta { delta: serde_json::to_string(&args).unwrap_or_default() };
                                    yield Event::ToolCallEnd { id: id.clone(), name: name.clone(), arguments: args.clone() };
                                    tool_calls.push((id, name, args, sig));
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
                                _ => StopReason::Error,
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

        if let Some(evt) = parser.finish()
            && evt.event == sse::EVENT_ERROR {
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                        format!("SSE error: {}", evt.data),
                    )),
                    message: Some(partial.clone()),
                };
                return;
            }

        if !current_thinking.is_empty() {
            partial.content.push(ContentBlock::Thinking {
                thinking: current_thinking,
                thinking_signature: current_thinking_signature,
                redacted: false,
            });
        }
        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text {
                text: current_text,
                text_signature: None,
            });
        }
        for (id, name, args, sig) in tool_calls {
            let arguments = match args {
                serde_json::Value::Object(map) => map.into_iter().collect(),
                _ => std::collections::HashMap::new(),
            };
            partial.content.push(ContentBlock::ToolCall {
                id,
                name,
                arguments,
                thought_signature: sig,
            });
        }
        if let Some(ref mut u) = partial.usage {
            crate::simple_options::finalize_usage(model, u);
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
    let mut contents: Vec<Value> = Vec::new();

    let transformed_messages = crate::transform::transform_messages(&context.messages, model);

    for msg in &transformed_messages {
        match msg.role {
            Role::ToolResult => {
                // Tool results must be sent as functionResponse parts, and consecutive
                // tool results must be merged into a single user turn (Cloud Code Assist).
                let text_result = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                let has_images = model.input.iter().any(|i| i == "image")
                    && msg.content.iter().any(|b| matches!(b, ContentBlock::Image { .. }));
                let response_value = if !text_result.is_empty() {
                    text_result
                } else if has_images {
                    "(see attached image)".to_string()
                } else {
                    String::new()
                };
                let response = if msg.is_error {
                    json!({"error": response_value})
                } else {
                    json!({"output": response_value})
                };
                let function_response_part = json!({
                    "functionResponse": {
                        "name": msg.tool_name.clone().unwrap_or_default(),
                        "response": response,
                    }
                });

                let merge = contents.last()
                    .and_then(|c| c.get("role").and_then(|r| r.as_str()).map(|r| r == "user")
                        .map(|is_user| is_user && c.get("parts").and_then(|p| p.as_array())
                            .map(|parts| parts.iter().any(|p| p.get("functionResponse").is_some()))
                            .unwrap_or(false)))
                    .unwrap_or(false);
                if merge {
                    if let Some(parts) = contents.last_mut().and_then(|c| c.get_mut("parts")).and_then(|p| p.as_array_mut()) {
                        parts.push(function_response_part);
                    }
                } else {
                    contents.push(json!({"role": "user", "parts": [function_response_part]}));
                }
            }
            Role::User => {
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
                if parts.is_empty() { continue; }
                contents.push(json!({"role": "user", "parts": parts}));
            }
            Role::Assistant => {
                let parts: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, text_signature } if !text.trim().is_empty() => {
                        let mut p = json!({"text": text});
                        if let Some(sig) = text_signature { p["thoughtSignature"] = json!(sig); }
                        Some(p)
                    }
                    ContentBlock::Image { data, mime_type } => Some(json!({
                        "inlineData": {"mimeType": mime_type, "data": data}
                    })),
                    ContentBlock::Thinking { thinking, thinking_signature, .. } if !thinking.trim().is_empty() => {
                        let mut p = json!({"thought": true, "text": thinking});
                        if let Some(sig) = thinking_signature { p["thoughtSignature"] = json!(sig); }
                        Some(p)
                    }
                    ContentBlock::ToolCall { name, arguments, thought_signature, .. } => {
                        let mut p = json!({"functionCall": {"name": name, "args": arguments}});
                        if let Some(sig) = thought_signature { p["thoughtSignature"] = json!(sig); }
                        Some(p)
                    }
                    _ => None,
                }).collect();
                if parts.is_empty() { continue; }
                contents.push(json!({"role": "model", "parts": parts}));
            }
        }
    }

    let mut payload = json!({"contents": contents});

    if let Some(ref prompt) = context.system_prompt {
        payload["systemInstruction"] = json!({"parts": [{"text": prompt}]});
    }

    let mut config = json!({});
    if let Some(max) = opts.max_tokens {
        config["maxOutputTokens"] = json!(max);
    }
    if let Some(temp) = opts.temperature {
        config["temperature"] = json!(temp);
    }
    // Thinking config for reasoning models.
    if model.reasoning {
        let id = model.id.to_lowercase();
        let is_gemini3_pro = id.contains("gemini-3") && id.contains("-pro");
        let is_gemini3_flash = id.contains("gemini-3") && id.contains("-flash");
        let is_gemma4 = id.contains("gemma-4") || id.contains("gemma4");
        if let Some(level) = opts.reasoning.as_ref() {
            let mut thinking_config = json!({"includeThoughts": true});
            let effort = format!("{:?}", level).to_lowercase();
            if is_gemini3_pro || is_gemini3_flash || is_gemma4 {
                // Gemini 3 / Gemma 4 use a thinkingLevel string instead of a token budget.
                let tl = if is_gemini3_pro {
                    match effort.as_str() { "minimal" | "low" => "LOW", _ => "HIGH" }
                } else if is_gemma4 {
                    match effort.as_str() { "minimal" | "low" => "MINIMAL", _ => "HIGH" }
                } else {
                    match effort.as_str() { "minimal" => "MINIMAL", "low" => "LOW", "medium" => "MEDIUM", _ => "HIGH" }
                };
                thinking_config["thinkingLevel"] = json!(tl);
            } else if let Some(budget) = opts.thinking_budgets.as_ref()
                .and_then(|b| b.medium.or(b.high).or(b.low).or(b.minimal)) {
                thinking_config["thinkingBudget"] = json!(budget);
            }
            config["thinkingConfig"] = thinking_config;
        } else {
            // Reasoning not requested: explicitly disable thinking (mirrors getDisabledThinkingConfig).
            let disabled = if is_gemini3_pro {
                json!({"thinkingLevel": "LOW"})
            } else if is_gemini3_flash || is_gemma4 {
                json!({"thinkingLevel": "MINIMAL"})
            } else {
                json!({"thinkingBudget": 0})
            };
            config["thinkingConfig"] = disabled;
        }
    }
    if config != json!({}) {
        payload["generationConfig"] = config;
    }

    if !context.tools.is_empty() {
        let decls: Vec<Value> = context.tools.iter().map(|t| {
            json!({"name": t.name, "description": t.description, "parameters": t.parameters})
        }).collect();
        payload["tools"] = json!([{"functionDeclarations": decls}]);

        // Tool choice -> functionCallingConfig mode.
        if let Some(ref tc) = opts.tool_choice {
            let mode = match tc.as_str() {
                Some("auto") => "AUTO",
                Some("any") | Some("required") => "ANY",
                Some("none") => "NONE",
                _ => "AUTO",
            };
            payload["toolConfig"] = json!({"functionCallingConfig": {"mode": mode}});
        }
    }

    payload
}

fn url_encode(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('+', "%2B")
        .replace('/', "%2F")
}
