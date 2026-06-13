//! Anthropic Messages API provider.

use std::sync::Arc;

use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start an Anthropic Messages stream.
pub fn stream_anthropic<'a>(
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

    let payload = build_anthropic_payload(model, context, opts);
    let url = format!("{}/messages", model.base_url.trim_end_matches('/'));

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("x-api-key", HeaderValue::from_str(&api_key).unwrap());
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

    // Beta features (prompt caching is GA and no longer requires a beta header).
    let mut beta_features: Vec<&str> = Vec::new();
    let is_fireworks = model.provider == "fireworks" || model.base_url.contains("fireworks.ai");
    if !context.tools.is_empty() && is_fireworks {
        beta_features.push("fine-grained-tool-streaming-2025-05-14");
    }
    if opts.reasoning.is_some() && model.reasoning {
        beta_features.push("interleaved-thinking-2025-05-14");
    }
    if !beta_features.is_empty()
        && let Ok(val) = HeaderValue::from_str(&beta_features.join(",")) {
            headers.insert("anthropic-beta", val);
    }

    // Session affinity header for providers that require it (Fireworks / Cloudflare AI Gateway).
    if let Some(ref session_id) = opts.session_id {
        let needs_affinity = model.provider == "fireworks"
            || model.base_url.contains("fireworks.ai")
            || model.base_url.contains("gateway.ai.cloudflare.com");
        if needs_affinity
            && let Ok(val) = HeaderValue::from_str(session_id) {
                headers.insert("x-session-affinity", val);
            }
    }

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
        let mut current_block_type = String::new();
        let mut current_thinking = String::new();
        let mut current_thinking_signature: Option<String> = None;
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_args = String::new();

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

                let data: Value = match serde_json::from_str(&evt.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = evt.event.as_str();
                match event_type {
                    "message_start" => {
                        if let Some(id) = data.pointer("/message/id").and_then(|v| v.as_str()) {
                            partial.response_id = Some(id.to_string());
                        }
                        if let Some(model_name) = data.pointer("/message/model").and_then(|v| v.as_str()) {
                            partial.response_model = Some(model_name.to_string());
                        }
                        if let Some(usage) = data.pointer("/message/usage") {
                            partial.usage = Some(parse_anthropic_usage(usage));
                        }
                    }
                    "content_block_start" => {
                        let block_type = data.pointer("/content_block/type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        current_block_type = block_type.clone();
                        match block_type.as_str() {
                            "text" => {
                                text_started = true;
                                current_text.clear();
                                yield Event::TextStart;
                            }
                            "thinking" => {
                                current_thinking.clear();
                                current_thinking_signature = None;
                                yield Event::ThinkingStart;
                            }
                            "tool_use" => {
                                current_tool_id = data.pointer("/content_block/id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                current_tool_name = data.pointer("/content_block/name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                current_tool_args.clear();
                                yield Event::ToolCallStart { id: current_tool_id.clone(), name: current_tool_name.clone() };
                            }
                            _ => {}
                        }
                    }
                    "content_block_delta" => {
                        let delta_type = data.pointer("/delta/type").and_then(|v| v.as_str()).unwrap_or("");
                        match delta_type {
                            "text_delta" => {
                                if let Some(text) = data.pointer("/delta/text").and_then(|v| v.as_str()) {
                                    current_text.push_str(text);
                                    yield Event::TextDelta { delta: text.to_string() };
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) = data.pointer("/delta/thinking").and_then(|v| v.as_str()) {
                                    current_thinking.push_str(thinking);
                                    yield Event::ThinkingDelta { delta: thinking.to_string() };
                                }
                            }
                            "signature_delta" => {
                                if let Some(sig) = data.pointer("/delta/signature").and_then(|v| v.as_str()) {
                                    current_thinking_signature = Some(sig.to_string());
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial_json) = data.pointer("/delta/partial_json").and_then(|v| v.as_str()) {
                                    current_tool_args.push_str(partial_json);
                                    yield Event::ToolCallDelta { delta: partial_json.to_string() };
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        match current_block_type.as_str() {
                            "text" => {
                                if text_started {
                                    text_started = false;
                                    yield Event::TextEnd;
                                }
                                if !current_text.is_empty() {
                                    partial.content.push(ContentBlock::Text {
                                        text: std::mem::take(&mut current_text),
                                        text_signature: None,
                                    });
                                }
                            }
                            "thinking" => {
                                yield Event::ThinkingEnd;
                                partial.content.push(ContentBlock::Thinking {
                                    thinking: std::mem::take(&mut current_thinking),
                                    thinking_signature: current_thinking_signature.take(),
                                    redacted: false,
                                });
                            }
                            "tool_use" => {
                                let parsed: Value = crate::jsonparse::parse_streaming_json(&current_tool_args);
                                let parsed_map = match &parsed {
                                    Value::Object(map) => map.clone().into_iter().collect(),
                                    _ => std::collections::HashMap::new(),
                                };
                                partial.content.push(ContentBlock::ToolCall {
                                    id: current_tool_id.clone(),
                                    name: current_tool_name.clone(),
                                    arguments: parsed_map,
                                    thought_signature: None,
                                });
                                yield Event::ToolCallEnd {
                                    id: std::mem::take(&mut current_tool_id),
                                    name: std::mem::take(&mut current_tool_name),
                                    arguments: parsed,
                                };
                                current_tool_args.clear();
                            }
                            _ => {}
                        }
                        current_block_type.clear();
                    }
                    "message_delta" => {
                        if let Some(reason) = data.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                            let stop_details = data.pointer("/delta/stop_details");
                            partial.stop_reason = Some(match reason {
                                "end_turn" => StopReason::Stop,
                                "max_tokens" => StopReason::Length,
                                "tool_use" => StopReason::ToolUse,
                                "pause_turn" => StopReason::Stop,
                                "stop_sequence" => StopReason::Stop,
                                "refusal" => {
                                    let explanation = stop_details
                                        .and_then(|d| d.get("explanation"))
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| "The model refused to complete the request".to_string());
                                    partial.error_message = Some(explanation);
                                    StopReason::Error
                                }
                                "sensitive" => StopReason::Error,
                                _ => StopReason::Stop,
                            });
                        }
                        if let Some(usage) = data.get("usage")
                            && let Some(output) = usage.get("output_tokens").and_then(|v| v.as_u64())
                                && let Some(ref mut u) = partial.usage {
                                    u.output = output as u32;
                                    u.total_tokens = u.input + u.output + u.cache_read + u.cache_write;
                                }
                    }
                    "message_stop" => {}
                    "error" => {
                        let msg = data.pointer("/error/message").and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "Anthropic stream error".to_string());
                        partial.stop_reason = Some(StopReason::Error);
                        partial.error_message = Some(msg.clone());
                        yield Event::Error {
                            reason: StopReason::Error,
                            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                            message: Some(partial.clone()),
                        };
                        return;
                    }
                    _ => {}
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

        if let Some(ref mut u) = partial.usage {

            crate::simple_options::finalize_usage(model, u);

        }

        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

pub(crate) fn build_anthropic_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    let transformed_messages = crate::transform::transform_messages(&context.messages, model);

    let mut i = 0usize;
    while i < transformed_messages.len() {
        let msg = &transformed_messages[i];
        if msg.role == Role::ToolResult {
            // Merge all consecutive tool-result messages into a single user message,
            // as Anthropic requires (and parallel tool calls produce multiple results).
            let mut tool_results: Vec<Value> = Vec::new();
            while i < transformed_messages.len() && transformed_messages[i].role == Role::ToolResult {
                let tr = &transformed_messages[i];
                let result_content: Vec<Value> = tr.content.iter().map(|b| match b {
                    ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
                    ContentBlock::Image { data, mime_type } => json!({
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime_type, "data": data}
                    }),
                    _ => json!({"type": "text", "text": ""}),
                }).collect();
                let mut tool_result = json!({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id.clone().unwrap_or_default(),
                    "content": result_content,
                });
                if tr.is_error {
                    tool_result["is_error"] = json!(true);
                }
                tool_results.push(tool_result);
                i += 1;
            }
            messages.push(json!({"role": "user", "content": tool_results}));
            continue;
        }

        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::ToolResult => unreachable!(),
        };
        let content: Vec<Value> = msg.content.iter().map(|b| match b {
            ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
            ContentBlock::Image { data, mime_type } => json!({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": data}
            }),
            ContentBlock::Thinking { thinking, thinking_signature, .. } => {
                let mut block = json!({"type": "thinking", "thinking": thinking});
                if let Some(sig) = thinking_signature {
                    block["signature"] = json!(sig);
                }
                block
            }
            ContentBlock::ToolCall { id, name, arguments, .. } => json!({
                "type": "tool_use", "id": id, "name": name, "input": arguments
            }),
        }).collect();
        messages.push(json!({"role": role_str, "content": content}));
        i += 1;
    }

    // Cache control (ephemeral) when prompt caching is enabled.
    let cache_control: Option<Value> = match opts.cache_retention {
        Some(CacheRetention::None) | None => None,
        Some(CacheRetention::Short) => Some(json!({"type": "ephemeral"})),
        Some(CacheRetention::Long) => Some(json!({"type": "ephemeral", "ttl": "1h"})),
    };

    // Apply cache_control to the last content block of the last message.
    if let Some(ref cc) = cache_control
        && let Some(last_msg) = messages.last_mut()
            && let Some(blocks) = last_msg.get_mut("content").and_then(|c| c.as_array_mut())
                && let Some(last_block) = blocks.last_mut() {
                    last_block["cache_control"] = cc.clone();
                }

    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true,
        "max_tokens": opts.max_tokens.unwrap_or(model.max_tokens),
    });

    if let Some(ref prompt) = context.system_prompt {
        // Structured system prompt allows attaching cache_control.
        let mut system_block = json!({"type": "text", "text": prompt});
        if let Some(ref cc) = cache_control {
            system_block["cache_control"] = cc.clone();
        }
        payload["system"] = json!([system_block]);
    }

    // Temperature is incompatible with extended thinking.
    let thinking_enabled = opts.reasoning.is_some() && model.reasoning;
    if let Some(temp) = opts.temperature
        && !thinking_enabled {
            payload["temperature"] = json!(temp);
        }

    // Thinking/reasoning support
    if thinking_enabled {
        let budget = opts.thinking_budgets.as_ref().and_then(|b| b.medium.or(b.high).or(b.low).or(b.minimal)).unwrap_or(8192);
        payload["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
    }

    if !context.tools.is_empty() {
        let mut tools: Vec<Value> = context.tools.iter().map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        }).collect();
        // Cache control on the last tool definition.
        if let Some(ref cc) = cache_control
            && let Some(last) = tools.last_mut() {
                last["cache_control"] = cc.clone();
            }
        payload["tools"] = json!(tools);
    }

    // Tool choice: a bare string becomes {type: string}; objects pass through.
    if let Some(ref tc) = opts.tool_choice {
        if let Some(s) = tc.as_str() {
            payload["tool_choice"] = json!({"type": s});
        } else {
            payload["tool_choice"] = tc.clone();
        }
    }

    // Metadata: only user_id is forwarded (mirrors upstream).
    if let Some(ref metadata) = opts.metadata
        && let Some(user_id) = metadata.get("user_id").and_then(|v| v.as_str()) {
            payload["metadata"] = json!({"user_id": user_id});
        }

    payload
}

fn parse_anthropic_usage(usage: &Value) -> Usage {
    Usage {
        input: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        output: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        cache_read: usage.get("cache_read_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        cache_write: usage.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        total_tokens: 0,
        cost: CostBreakdown::default(),
    }
}
