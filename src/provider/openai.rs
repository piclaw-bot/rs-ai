//! OpenAI Chat Completions provider (also serves compatible APIs).

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
    let mut payload = build_payload(model, context, opts, &compat);
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

    let url = format!("{}/chat/completions", crate::utils::resolve_cloudflare_base_url(model.base_url.trim_end_matches('/')).trim_end_matches('/'));
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("Accept", HeaderValue::from_static("text/event-stream"));

    if model.provider == "cloudflare-ai-gateway" {
        headers.insert("cf-aig-authorization", HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    } else {
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    }

    // GitHub Copilot dynamic headers (mirrors upstream buildCopilotDynamicHeaders)
    if model.provider == "github-copilot" {
        let initiator = match context.messages.last() {
            Some(m) if m.role != Role::User => "agent",
            _ => "user",
        };
        headers.insert("X-Initiator", HeaderValue::from_static(if initiator == "agent" { "agent" } else { "user" }));
        headers.insert("Openai-Intent", HeaderValue::from_static("conversation-edits"));
        let has_images = context.messages.iter().any(|m| {
            matches!(m.role, Role::User | Role::ToolResult)
                && m.content.iter().any(|c| matches!(c, ContentBlock::Image { .. }))
        });
        if has_images {
            headers.insert("Copilot-Vision-Request", HeaderValue::from_static("true"));
        }
    }

    // Session affinity headers for providers that require them.
    if let Some(ref session_id) = opts.session_id {
        if compat.supports_session_affinity_headers == Some(true) {
            if let Ok(val) = HeaderValue::from_str(session_id) {
                headers.insert("session_id", val.clone());
                headers.insert("x-client-request-id", val.clone());
                headers.insert("x-session-affinity", val);
            }
        }
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

        let mut text_started = false;
        let mut current_text = String::new();
        let mut thinking_started = false;
        let mut current_thinking = String::new();
        let mut current_thinking_signature: Option<String> = None;
        let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> = std::collections::BTreeMap::new();

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

                        let reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"];
                        for field in reasoning_fields {
                            if let Some(reasoning) = delta.get(field).and_then(|v| v.as_str()) {
                                if !reasoning.is_empty() {
                                    if !thinking_started {
                                        thinking_started = true;
                                        current_thinking_signature = Some(field.to_string());
                                        yield Event::ThinkingStart;
                                    }
                                    current_thinking.push_str(reasoning);
                                    yield Event::ThinkingDelta { delta: reasoning.to_string() };
                                    break;
                                }
                            }
                        }

                        if let Some(delta_tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc in delta_tool_calls {
                                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                let entry = tool_calls.entry(index).or_insert_with(|| (String::new(), String::new(), String::new()));
                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    if entry.0.is_empty() {
                                        entry.0 = id.to_string();
                                    }
                                }
                                if let Some(func) = tc.get("function") {
                                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                        if entry.1.is_empty() {
                                            entry.1 = name.to_string();
                                            if !entry.1.is_empty() {
                                                yield Event::ToolCallStart { id: entry.0.clone(), name: entry.1.clone() };
                                            }
                                        }
                                    }
                                    if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                        if !args.is_empty() {
                                            entry.2.push_str(args);
                                            yield Event::ToolCallDelta { delta: args.to_string() };
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                            if text_started {
                                yield Event::TextEnd;
                                text_started = false;
                            }
                            if thinking_started {
                                yield Event::ThinkingEnd;
                                thinking_started = false;
                            }
                            let stop = match reason {
                                "stop" => StopReason::Stop,
                                "length" => StopReason::Length,
                                "tool_calls" => StopReason::ToolUse,
                                _ => StopReason::Stop,
                            };
                            partial.stop_reason = Some(stop.clone());
                            if !current_text.is_empty() && !partial.content.iter().any(|b| matches!(b, ContentBlock::Text { .. })) {
                                partial.content.push(ContentBlock::Text {
                                    text: current_text.clone(),
                                    text_signature: None,
                                });
                            }
                            if !current_thinking.is_empty() && !partial.content.iter().any(|b| matches!(b, ContentBlock::Thinking { .. })) {
                                partial.content.push(ContentBlock::Thinking {
                                    thinking: current_thinking.clone(),
                                    thinking_signature: current_thinking_signature.clone(),
                                    redacted: false,
                                });
                            }
                            if partial.content.iter().all(|b| !matches!(b, ContentBlock::ToolCall { .. })) {
                                for (_idx, (id, name, args_json)) in &tool_calls {
                                    let arguments = match crate::jsonparse::parse_streaming_json(args_json) {
                                        serde_json::Value::Object(map) => map.into_iter().collect(),
                                        _ => std::collections::HashMap::new(),
                                    };
                                    partial.content.push(ContentBlock::ToolCall {
                                        id: id.clone(),
                                        name: name.clone(),
                                        arguments,
                                        thought_signature: None,
                                    });
                                }
                            }
                        }
                    }
                }

                if let Some(usage) = chunk.get("usage") {
                    partial.usage = Some(Usage {
                        input: usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        output: usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
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

        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

pub(crate) fn build_payload(
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
    let transformed_messages = crate::transform::transform_messages(&context.messages, model);
    for msg in &transformed_messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::ToolResult => "tool",
        };

        let text_blocks: Vec<String> = msg.content.iter().filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        }).collect();
        let tool_call_blocks: Vec<Value> = msg.content.iter().filter_map(|b| match b {
            ContentBlock::ToolCall { id, name, arguments, .. } => Some(json!({
                "id": normalize_tool_call_id(id, &model.provider),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
                }
            })),
            _ => None,
        }).collect();

        let content: Value = if msg.role == Role::Assistant {
            // Collect non-empty thinking blocks for replay handling.
            let thinking_blocks: Vec<(&String, &Option<String>)> = msg.content.iter().filter_map(|b| match b {
                ContentBlock::Thinking { thinking, thinking_signature, .. } if !thinking.trim().is_empty() => Some((thinking, thinking_signature)),
                _ => None,
            }).collect();
            let assistant_text = if text_blocks.is_empty() { String::new() } else { text_blocks.join("") };

            if !thinking_blocks.is_empty() && compat.requires_thinking_as_text == Some(true) {
                // Convert thinking blocks into a leading text block (no tags).
                let thinking_text = thinking_blocks.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("\n\n");
                let mut parts = vec![json!({"type": "text", "text": thinking_text})];
                if !assistant_text.is_empty() {
                    parts.push(json!({"type": "text", "text": assistant_text}));
                }
                json!(parts)
            } else if assistant_text.is_empty() {
                Value::Null
            } else {
                json!(assistant_text)
            }
        } else if msg.content.len() == 1 {
            match &msg.content[0] {
                ContentBlock::Text { text, .. } => json!(text),
                _ => json!(format_content_blocks(&msg.content)),
            }
        } else {
            json!(format_content_blocks(&msg.content))
        };

        let mut m = json!({ "role": role_str, "content": content });
        if msg.role == Role::Assistant {
            if !tool_call_blocks.is_empty() {
                m["tool_calls"] = json!(tool_call_blocks);
            }
            // When not sending thinking-as-text, replay thinking via its signature field
            // (e.g. reasoning_content for llama.cpp / gpt-oss).
            if compat.requires_thinking_as_text != Some(true) {
                let thinking_blocks: Vec<(&String, &Option<String>)> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Thinking { thinking, thinking_signature, .. } if !thinking.trim().is_empty() => Some((thinking, thinking_signature)),
                    _ => None,
                }).collect();
                if let Some((_, Some(sig))) = thinking_blocks.first() {
                    if !sig.is_empty() {
                        let joined = thinking_blocks.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("\n");
                        m[sig.as_str()] = json!(joined);
                    }
                }
            }
            // DeepSeek-style providers require reasoning_content on assistant messages.
            if compat.requires_reasoning_content_on_assistant_messages == Some(true)
                && model.reasoning
                && m.get("reasoning_content").is_none() {
                m["reasoning_content"] = json!("");
            }
            // Skip empty assistant messages (no content and no tool calls).
            let has_content = !matches!(m["content"], Value::Null)
                && !(m["content"].as_str().map(|s| s.is_empty()).unwrap_or(false));
            if !has_content && tool_call_blocks.is_empty() {
                continue;
            }
        }
        if msg.role == Role::ToolResult {
            if let Some(ref id) = msg.tool_call_id {
                m["tool_call_id"] = json!(normalize_tool_call_id(id, &model.provider));
            }
            if compat.requires_tool_result_name == Some(true) {
                if let Some(ref name) = msg.tool_name {
                    m["name"] = json!(name);
                }
            }
        }
        messages.push(m);
    }

    let max_tokens_field = compat.max_tokens_field.as_deref().unwrap_or("max_completion_tokens");

    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true,
    });

    if compat.supports_usage_in_streaming != Some(false) {
        payload["stream_options"] = json!({ "include_usage": true });
    }
    if compat.supports_store == Some(true) {
        payload["store"] = json!(false);
    }

    if let Some(ref session_id) = opts.session_id {
        payload["sessionId"] = json!(session_id);
    }
    if let Some(ref metadata) = opts.metadata {
        payload["metadata"] = json!(metadata);
    }

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

    // Reasoning/thinking (clamped to the model's supported levels).
    // Mirrors upstream buildParams thinking-format handling, gated on model.reasoning.
    let clamped_effort = opts.reasoning.as_ref().and_then(|l| crate::simple_options::clamp_reasoning_for_model(model, l));
    if model.reasoning {
        let map_effort = |level: &ThinkingLevel| -> String {
            let key = format!("{:?}", level).to_lowercase();
            model.thinking_level_map.as_ref()
                .and_then(|m| m.get(&key))
                .and_then(|v| v.clone())
                .unwrap_or(key)
        };
        let off_value = || -> Option<String> {
            match model.thinking_level_map.as_ref().and_then(|m| m.get("off")) {
                Some(Some(s)) => Some(s.clone()),
                Some(None) => None,           // explicitly disabled
                None => Some("none".to_string()),
            }
        };
        match compat.thinking_format.as_deref() {
            Some("zai") => {
                payload["thinking"] = json!({"type": if clamped_effort.is_some() { "enabled" } else { "disabled" }});
            }
            Some("qwen") => {
                payload["enable_thinking"] = json!(clamped_effort.is_some());
            }
            Some("together") => {
                payload["reasoning"] = json!({"enabled": clamped_effort.is_some()});
                if let Some(ref level) = clamped_effort {
                    if compat.supports_reasoning_effort == Some(true) {
                        payload["reasoning_effort"] = json!(map_effort(level));
                    }
                }
            }
            Some("deepseek") => {
                payload["thinking"] = json!({"type": if clamped_effort.is_some() { "enabled" } else { "disabled" }});
                if let Some(ref level) = clamped_effort {
                    if compat.supports_reasoning_effort == Some(true) {
                        payload["reasoning_effort"] = json!(map_effort(level));
                    }
                }
            }
            Some("openrouter") => {
                if let Some(ref level) = clamped_effort {
                    payload["reasoning"] = json!({"effort": map_effort(level)});
                } else if let Some(off) = off_value() {
                    payload["reasoning"] = json!({"effort": off});
                }
            }
            Some("ant-ling") => {
                if let Some(ref level) = clamped_effort {
                    let key = format!("{:?}", level).to_lowercase();
                    if let Some(Some(mapped)) = model.thinking_level_map.as_ref().map(|m| m.get(&key).cloned().flatten()) {
                        payload["reasoning"] = json!({"effort": mapped});
                    }
                }
            }
            _ => {
                if compat.supports_reasoning_effort == Some(true) {
                    if let Some(ref level) = clamped_effort {
                        payload["reasoning_effort"] = json!(map_effort(level));
                    } else if let Some(Some(off)) = model.thinking_level_map.as_ref().map(|m| m.get("off").cloned().flatten()) {
                        payload["reasoning_effort"] = json!(off);
                    }
                }
            }
        }
    }

    // Tools
    if !context.tools.is_empty() {
        let include_strict = compat.supports_strict_mode != Some(false);
        let tools: Vec<Value> = context.tools.iter().map(|t| {
            let mut function = json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            });
            if include_strict {
                function["strict"] = json!(false);
            }
            json!({ "type": "function", "function": function })
        }).collect();
        payload["tools"] = json!(tools);
    }

    if let Some(ref tool_choice) = opts.tool_choice {
        payload["tool_choice"] = tool_choice.clone();
    }

    payload
}

/// Normalize a tool-call ID for OpenAI-compatible APIs (mirrors upstream `normalizeToolCallId`).
/// Pipe-separated IDs (from the Responses API) are reduced to the sanitized call_id,
/// and overly-long OpenAI IDs are truncated to 40 chars.
fn normalize_tool_call_id(id: &str, provider: &str) -> String {
    if let Some((call_id, _)) = id.split_once('|') {
        return call_id
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .take(40)
            .collect();
    }
    if provider == "openai" && id.len() > 40 {
        return id.chars().take(40).collect();
    }
    id.to_string()
}

fn format_content_blocks(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks.iter().map(|b| match b {
        ContentBlock::Text { text, .. } => json!({"type": "text", "text": text}),
        ContentBlock::Image { data, mime_type } => json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{};base64,{}", mime_type, data)}
        }),
        ContentBlock::Thinking { thinking, .. } => json!({"type": "text", "text": thinking}),
        ContentBlock::ToolCall { id: _, name, arguments, .. } => json!({
            "type": "text",
            "text": format!("[tool_call: {} {}]", name, serde_json::to_string(arguments).unwrap_or_default())
        }),
    }).collect()
}
