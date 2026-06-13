//! OpenAI Responses API provider (also serves Azure OpenAI Responses).

use std::sync::Arc;

use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::compat::detect_compat;
use crate::env::resolve_api_key;
use crate::events::Event;
use crate::simple_options::{adjust_max_tokens_for_thinking, default_thinking_budgets};
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

    let mut payload = build_responses_payload(model, context, opts);
    if let Some(ref hook) = opts.on_payload {
        match hook(payload.clone(), model) {
            Ok(next) => payload = next,
            Err(err) => {
                let err = Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(err),
                    message: None,
                };
                return Box::pin(stream::once(async { err }));
            }
        }
    }
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

    if let Some(ref extra_headers) = opts.headers {
        for (k, v) in extra_headers {
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

        let status = resp.status().as_u16();
        if let Some(ref hook) = opts.on_response {
            let mut hdrs = std::collections::HashMap::new();
            for (k, v) in resp.headers().iter() {
                hdrs.insert(k.to_string(), v.to_str().unwrap_or("").to_string());
            }
            hook(status, &hdrs, model);
        }

        if !resp.status().is_success() {
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
        let mut stream = resp.bytes_stream();

        let mut current_text = String::new();
        let mut text_started = false;
        let mut current_thinking = String::new();
        let mut current_tool_call_id: Option<String> = None;
        let mut current_tool_item_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_args = String::new();

        while let Some(chunk_result) = stream.next().await {
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

                let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match event_type {
                    "response.created" => {
                        if let Some(response) = data.get("response") {
                            if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                                partial.response_id = Some(id.to_string());
                            }
                            if let Some(model_name) = response.get("model").and_then(|v| v.as_str()) {
                                partial.response_model = Some(model_name.to_string());
                            }
                        }
                    }
                    "response.output_item.added" => {
                        if let Some(item) = data.get("item") {
                            match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                                "function_call" => {
                                    current_tool_call_id = item.get("call_id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                    current_tool_item_id = item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                                    current_tool_name = item.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
                                    current_tool_args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    if let (Some(id), Some(name)) = (current_tool_call_id.clone(), current_tool_name.clone()) {
                                        yield Event::ToolCallStart { id, name };
                                    }
                                    if !current_tool_args.is_empty() {
                                        yield Event::ToolCallDelta { delta: current_tool_args.clone() };
                                    }
                                }
                                "message" => {
                                    if !text_started {
                                        text_started = true;
                                        yield Event::TextStart;
                                    }
                                }
                                "reasoning" => {
                                    current_thinking.clear();
                                    yield Event::ThinkingStart;
                                }
                                _ => {}
                            }
                        }
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
                    "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                        if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                            current_thinking.push_str(delta);
                            yield Event::ThinkingDelta { delta: delta.to_string() };
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = data.get("delta").and_then(|v| v.as_str()) {
                            current_tool_args.push_str(delta);
                            yield Event::ToolCallDelta { delta: delta.to_string() };
                        }
                    }
                    "response.function_call_arguments.done" => {
                        if let Some(arguments) = data.get("arguments").and_then(|v| v.as_str()) {
                            if arguments.starts_with(&current_tool_args) {
                                let extra = &arguments[current_tool_args.len()..];
                                if !extra.is_empty() {
                                    current_tool_args.push_str(extra);
                                    yield Event::ToolCallDelta { delta: extra.to_string() };
                                }
                            } else {
                                current_tool_args = arguments.to_string();
                            }
                        }
                    }
                    "response.content_part.done" => {
                        if text_started {
                            text_started = false;
                            yield Event::TextEnd;
                        }
                    }
                    "response.output_item.done" => {
                        if let Some(item) = data.get("item") {
                            match item.get("type").and_then(|v| v.as_str()) {
                                Some("function_call") => {
                                    let id = item.get("call_id").and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                        .or_else(|| current_tool_call_id.clone())
                                        .unwrap_or_default();
                                    let name = item.get("name").and_then(|v| v.as_str())
                                        .map(|s| s.to_string())
                                        .or_else(|| current_tool_name.clone())
                                        .unwrap_or_default();
                                    let final_args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or(&current_tool_args);
                                    let parsed: serde_json::Value = crate::jsonparse::parse_streaming_json(final_args);
                                    let parsed_map = match &parsed {
                                        serde_json::Value::Object(map) => map.clone().into_iter().collect(),
                                        _ => std::collections::HashMap::new(),
                                    };
                                    partial.content.push(ContentBlock::ToolCall {
                                        id: match item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| current_tool_item_id.clone()) {
                                            Some(item_id) if !id.is_empty() => format!("{}|{}", id, item_id),
                                            _ => id.clone(),
                                        },
                                        name: name.clone(),
                                        arguments: parsed_map,
                                        thought_signature: None,
                                    });
                                    yield Event::ToolCallEnd {
                                        id,
                                        name,
                                        arguments: parsed,
                                    };
                                    current_tool_call_id = None;
                                    current_tool_item_id = None;
                                    current_tool_name = None;
                                    current_tool_args.clear();
                                }
                                Some("reasoning") => {
                                    let thinking_text = item.get("summary").and_then(|v| v.as_array())
                                        .map(|parts| parts.iter().filter_map(|p| p.get("text").and_then(|v| v.as_str())).collect::<Vec<_>>().join("\n\n"))
                                        .filter(|s| !s.is_empty())
                                        .or_else(|| item.get("content").and_then(|v| v.as_array())
                                            .map(|parts| parts.iter().filter_map(|p| p.get("text").and_then(|v| v.as_str())).collect::<Vec<_>>().join("\n\n"))
                                            .filter(|s| !s.is_empty()))
                                        .unwrap_or_else(|| current_thinking.clone());
                                    partial.content.push(ContentBlock::Thinking {
                                        thinking: thinking_text,
                                        thinking_signature: Some(item.to_string()),
                                        redacted: false,
                                    });
                                    yield Event::ThinkingEnd;
                                    current_thinking.clear();
                                }
                                _ => {}
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(response) = data.get("response") {
                            if let Some(model_name) = response.get("model").and_then(|v| v.as_str()) {
                                partial.response_model = Some(model_name.to_string());
                            }
                            if let Some(usage) = response.get("usage") {
                                partial.usage = Some(Usage {
                                    input: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    output: usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    ..Default::default()
                                });
                            }
                            partial.stop_reason = Some(if partial.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. })) {
                                StopReason::ToolUse
                            } else {
                                StopReason::Stop
                            });
                        }
                    }
                    _ => {}
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

pub(crate) fn build_responses_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let mut input = Vec::new();

    if let Some(ref prompt) = context.system_prompt {
        input.push(json!({"role": "system", "content": prompt}));
    }

    let transformed_messages = crate::transform::transform_messages(&context.messages, model);

    for msg in &transformed_messages {
        match msg.role {
            Role::User => {
                if msg.content.len() == 1 {
                    match &msg.content[0] {
                        ContentBlock::Text { text, .. } => input.push(json!({"role": "user", "content": text})),
                        ContentBlock::Image { data, mime_type } => input.push(json!({
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": format!("data:{};base64,{}", mime_type, data)}]
                        })),
                        _ => {}
                    }
                } else {
                    let parts: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(json!({"type": "input_text", "text": text})),
                        ContentBlock::Image { data, mime_type } => Some(json!({
                            "type": "input_image", "image_url": format!("data:{};base64,{}", mime_type, data)
                        })),
                        _ => None,
                    }).collect();
                    input.push(json!({"role": "user", "content": parts}));
                }
            }
            Role::Assistant => {
                let text_parts: Vec<String> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }).collect();
                if !text_parts.is_empty() {
                    input.push(json!({"role": "assistant", "content": text_parts.join("\n")}));
                }
                for block in &msg.content {
                    match block {
                        ContentBlock::Thinking { thinking_signature: Some(sig), .. } => {
                            if let Ok(v) = serde_json::from_str::<Value>(sig) {
                                input.push(v);
                            }
                        }
                        ContentBlock::ToolCall { id, name, arguments, .. } => {
                            let (call_id, item_id) = id.split_once('|').map(|(a,b)| (a.to_string(), Some(b.to_string()))).unwrap_or((id.clone(), None));
                            input.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": name,
                                "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
                            }));
                        }
                        _ => {}
                    }
                }
            }
            Role::ToolResult => {
                let text_result = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                let image_parts: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Image { data, mime_type } => Some(json!({
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": format!("data:{};base64,{}", mime_type, data)
                    })),
                    _ => None,
                }).collect();
                let call_id = msg.tool_call_id.as_deref().and_then(|id| id.split('|').next()).unwrap_or_default();
                if !image_parts.is_empty() {
                    let mut output = Vec::new();
                    if !text_result.is_empty() {
                        output.push(json!({"type": "input_text", "text": text_result}));
                    }
                    output.extend(image_parts);
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }));
                } else {
                    input.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": text_result,
                    }));
                }
            }
        }
    }

    let compat = detect_compat(model);
    let mut payload = json!({
        "model": model.id,
        "input": input,
        "stream": true,
        "store": false,
    });

    if let Some(ref session_id) = opts.session_id {
        payload["session_id"] = json!(session_id);
    }
    if let Some(ref previous_response_id) = opts.previous_response_id {
        payload["previous_response_id"] = json!(previous_response_id);
    }
    if let Some(ref metadata) = opts.metadata {
        payload["metadata"] = json!(metadata);
    }

    match opts.cache_retention {
        Some(CacheRetention::None) => {}
        Some(CacheRetention::Short) => {
            if let Some(ref session_id) = opts.session_id {
                payload["prompt_cache_key"] = json!(session_id);
            }
        }
        Some(CacheRetention::Long) => {
            if let Some(ref session_id) = opts.session_id {
                payload["prompt_cache_key"] = json!(session_id);
            }
            if compat.supports_long_cache_retention != Some(false) {
                payload["prompt_cache_retention"] = json!("24h");
            }
        }
        None => {}
    }

    let mut effective_max_tokens = opts.max_tokens.unwrap_or(model.max_tokens);
    if let Some(ref level) = opts.reasoning {
        let budgets_map = if let Some(ref budgets) = opts.thinking_budgets {
            let mut map = default_thinking_budgets();
            if let Some(v) = budgets.minimal { map.insert(ThinkingLevel::Minimal, v); }
            if let Some(v) = budgets.low { map.insert(ThinkingLevel::Low, v); }
            if let Some(v) = budgets.medium { map.insert(ThinkingLevel::Medium, v); }
            if let Some(v) = budgets.high { map.insert(ThinkingLevel::High, v); }
            map
        } else {
            default_thinking_budgets()
        };
        let (adjusted_max, _budget) = adjust_max_tokens_for_thinking(effective_max_tokens, model.max_tokens, level, &budgets_map);
        effective_max_tokens = adjusted_max;
    }
    if effective_max_tokens > 0 {
        payload["max_output_tokens"] = json!(effective_max_tokens);
    }
    if let Some(temp) = opts.temperature {
        payload["temperature"] = json!(temp);
    }

    if let Some(level) = opts.reasoning.as_ref().and_then(|l| crate::simple_options::clamp_reasoning_for_model(model, l)) {
        payload["reasoning"] = json!({
            "effort": format!("{:?}", level).to_lowercase(),
            "summary": opts.reasoning_summary.clone().unwrap_or_else(|| "auto".to_string()),
        });
        payload["include"] = json!(["reasoning.encrypted_content"]);
    }

    if !context.tools.is_empty() {
        let tools: Vec<Value> = context.tools.iter().map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": false,
            })
        }).collect();
        payload["tools"] = json!(tools);
    }

    payload
}
