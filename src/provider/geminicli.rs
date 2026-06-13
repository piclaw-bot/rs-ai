//! Google Gemini CLI (Cloud Code Assist) provider.
//!
//! Uses the same Google Generative AI SSE protocol but with different
//! authentication (OAuth token via Gemini CLI flow).

use std::sync::Arc;

use futures::stream;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::Value;

use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start a Gemini CLI stream.
pub fn stream_geminicli<'a>(
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

    let mut payload = build_geminicli_payload(model, context, opts);
    if let Some(ref hook) = opts.on_payload {
        match hook(payload.clone(), model) {
            Ok(next) => payload = next,
            Err(err) => {
                let err = Event::Error { reason: StopReason::Error, error: Arc::from(err), message: None };
                return Box::pin(stream::once(async { err }));
            }
        }
    }
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        model.base_url.trim_end_matches('/'),
        url_encode(&model.id),
    );

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("accept", HeaderValue::from_static("text/event-stream"));
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
        let request = client.post(&url).headers(headers).json(&payload);
        let request = if let Some(ms) = opts.timeout_ms {
            request.timeout(std::time::Duration::from_millis(ms))
        } else { request };
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

        // Invoke the on_response hook with the status + headers (mirrors options.onResponse).
        if let Some(ref hook) = opts.on_response {
            let status = resp.status().as_u16();
            let mut hdrs = std::collections::HashMap::new();
            for (k, v) in resp.headers().iter() {
                hdrs.insert(k.to_string(), v.to_str().unwrap_or("").to_string());
            }
            hook(status, &hdrs, model);
        }

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
        let mut current_text_signature: Option<String> = None;
        let mut current_thinking = String::new();
        let mut current_thinking_signature: Option<String> = None;
        // Streaming block state: 0 = none, 1 = text, 2 = thinking (preserves interleaving).
        let mut block_kind: u8 = 0;
        let mut tool_call_ids: Vec<String> = Vec::new();

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
                                let part_sig = part.get("thoughtSignature").and_then(|v| v.as_str()).filter(|s| !s.is_empty());
                                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                                    && !text.is_empty() {
                                    let want: u8 = if is_thought { 2 } else { 1 };
                                    if block_kind != want {
                                        match block_kind {
                                            1 => {
                                                yield Event::TextEnd;
                                                partial.content.push(ContentBlock::Text {
                                                    text: std::mem::take(&mut current_text),
                                                    text_signature: current_text_signature.take(),
                                                });
                                            }
                                            2 => {
                                                yield Event::ThinkingEnd;
                                                partial.content.push(ContentBlock::Thinking {
                                                    thinking: std::mem::take(&mut current_thinking),
                                                    thinking_signature: current_thinking_signature.take(),
                                                    redacted: false,
                                                });
                                            }
                                            _ => {}
                                        }
                                        if want == 2 { yield Event::ThinkingStart; } else { yield Event::TextStart; }
                                        block_kind = want;
                                    }
                                    if is_thought {
                                        current_thinking.push_str(text);
                                        if let Some(sig) = part_sig { current_thinking_signature = Some(sig.to_string()); }
                                        yield Event::ThinkingDelta { delta: text.to_string() };
                                    } else {
                                        current_text.push_str(text);
                                        if let Some(sig) = part_sig { current_text_signature = Some(sig.to_string()); }
                                        yield Event::TextDelta { delta: text.to_string() };
                                    }
                                }
                                if let Some(fc) = part.get("functionCall") {
                                    match block_kind {
                                        1 => {
                                            yield Event::TextEnd;
                                            partial.content.push(ContentBlock::Text {
                                                text: std::mem::take(&mut current_text),
                                                text_signature: current_text_signature.take(),
                                            });
                                        }
                                        2 => {
                                            yield Event::ThinkingEnd;
                                            partial.content.push(ContentBlock::Thinking {
                                                thinking: std::mem::take(&mut current_thinking),
                                                thinking_signature: current_thinking_signature.take(),
                                                redacted: false,
                                            });
                                        }
                                        _ => {}
                                    }
                                    block_kind = 0;
                                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let args = fc.get("args").cloned().unwrap_or_else(|| serde_json::json!({}));
                                    let provided = fc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                    let needs_new = match &provided {
                                        None => true,
                                        Some(pid) => tool_call_ids.iter().any(|eid| eid == pid),
                                    };
                                    let id = if needs_new {
                                        format!("{}_{}_{}", name, crate::utils::now_millis(), tool_call_ids.len() + 1)
                                    } else {
                                        provided.unwrap()
                                    };
                                    let sig = part_sig.map(|s| s.to_string());
                                    yield Event::ToolCallStart { id: id.clone(), name: name.clone() };
                                    yield Event::ToolCallDelta { delta: serde_json::to_string(&args).unwrap_or_default() };
                                    yield Event::ToolCallEnd { id: id.clone(), name: name.clone(), arguments: args.clone() };
                                    let arguments = match &args {
                                        serde_json::Value::Object(map) => map.clone().into_iter().collect(),
                                        _ => std::collections::HashMap::new(),
                                    };
                                    partial.content.push(ContentBlock::ToolCall {
                                        id: id.clone(), name, arguments, thought_signature: sig,
                                    });
                                    tool_call_ids.push(id);
                                }
                            }
                        }
                        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str()) {
                            match block_kind {
                                1 => {
                                    yield Event::TextEnd;
                                    partial.content.push(ContentBlock::Text {
                                        text: std::mem::take(&mut current_text),
                                        text_signature: current_text_signature.take(),
                                    });
                                }
                                2 => {
                                    yield Event::ThinkingEnd;
                                    partial.content.push(ContentBlock::Thinking {
                                        thinking: std::mem::take(&mut current_thinking),
                                        thinking_signature: current_thinking_signature.take(),
                                        redacted: false,
                                    });
                                }
                                _ => {}
                            }
                            block_kind = 0;
                            partial.stop_reason = Some(match reason {
                                "STOP" if !tool_call_ids.is_empty() => StopReason::ToolUse,
                                "STOP" => StopReason::Stop,
                                "MAX_TOKENS" => StopReason::Length,
                                other => {
                                    partial.error_message = Some(format!("Gemini stopped with finish reason: {other}"));
                                    StopReason::Error
                                }
                            });
                        }
                    }
                }

                if let Some(usage) = chunk.get("usageMetadata") {
                    let prompt = usage.get("promptTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let cached = usage.get("cachedContentTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let candidates = usage.get("candidatesTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    let thoughts = usage.get("thoughtsTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    partial.usage = Some(Usage {
                        input: prompt.saturating_sub(cached),
                        output: candidates + thoughts,
                        cache_read: cached,
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

        // Finalize any block still open when the stream ends without a finishReason.
        match block_kind {
            1 if !current_text.is_empty() => {
                partial.content.push(ContentBlock::Text {
                    text: std::mem::take(&mut current_text),
                    text_signature: current_text_signature.take(),
                });
            }
            2 if !current_thinking.is_empty() => {
                partial.content.push(ContentBlock::Thinking {
                    thinking: std::mem::take(&mut current_thinking),
                    thinking_signature: current_thinking_signature.take(),
                    redacted: false,
                });
            }
            _ => {}
        }
        if let Some(ref mut u) = partial.usage {
            crate::simple_options::finalize_usage(model, u);
        }
        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        if matches!(reason, StopReason::Error | StopReason::Aborted) {
            let msg = partial.error_message.clone().unwrap_or_else(|| "An unknown error occurred".to_string());
            yield Event::Error {
                reason,
                error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                message: Some(partial),
            };
        } else {
            yield Event::Done { reason, message: partial };
        }
    })
}

fn build_geminicli_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    // Same payload format as Google Generative AI
    crate::provider::google::build_google_payload_public(model, context, opts)
}

fn url_encode(s: &str) -> String {
    s.replace('%', "%25").replace(' ', "%20").replace('+', "%2B").replace('/', "%2F")
}
