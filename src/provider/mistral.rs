//! Mistral Conversations API provider.

use std::sync::Arc;

use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::env::resolve_api_key;
use crate::events::Event;
use crate::transports::sse;
use crate::types::*;

/// Start a Mistral stream (OpenAI-compatible with small differences).
pub fn stream_mistral<'a>(
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

    let payload = build_mistral_payload(model, context, opts);
    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
    headers.insert("Accept", HeaderValue::from_static("text/event-stream"));

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

        let mut text_started = false;
        let mut current_text = String::new();
        let mut thinking_started = false;
        let mut current_thinking = String::new();
        let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> = std::collections::BTreeMap::new();
        let mut got_done = false;

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
                if evt.data == "[DONE]" {
                    got_done = true;
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
                        if let Some(delta) = choice.get("delta") {
                            // Mistral `content` may be a string or an array of content items
                            // (text and thinking, used by magistral reasoning models).
                            match delta.get("content") {
                                Some(serde_json::Value::String(s)) if !s.is_empty() => {
                                    if !text_started {
                                        text_started = true;
                                        yield Event::TextStart;
                                    }
                                    current_text.push_str(s);
                                    yield Event::TextDelta { delta: s.clone() };
                                }
                                Some(serde_json::Value::Array(items)) => {
                                    for item in items {
                                        if let Some(s) = item.as_str() {
                                            if !s.is_empty() {
                                                if !text_started { text_started = true; yield Event::TextStart; }
                                                current_text.push_str(s);
                                                yield Event::TextDelta { delta: s.to_string() };
                                            }
                                            continue;
                                        }
                                        match item.get("type").and_then(|v| v.as_str()) {
                                            Some("text") => {
                                                if let Some(t) = item.get("text").and_then(|v| v.as_str())
                                                    && !t.is_empty() {
                                                        if !text_started { text_started = true; yield Event::TextStart; }
                                                        current_text.push_str(t);
                                                        yield Event::TextDelta { delta: t.to_string() };
                                                }
                                            }
                                            Some("thinking") => {
                                                // `thinking` is a string or an array of {type:text,text}.
                                                let tt = match item.get("thinking") {
                                                    Some(serde_json::Value::String(s)) => s.clone(),
                                                    Some(serde_json::Value::Array(parts)) => parts.iter()
                                                        .filter_map(|p| p.get("text").and_then(|v| v.as_str()).or_else(|| p.as_str()))
                                                        .collect::<Vec<_>>().join(""),
                                                    _ => String::new(),
                                                };
                                                if !tt.is_empty() {
                                                    if !thinking_started { thinking_started = true; yield Event::ThinkingStart; }
                                                    current_thinking.push_str(&tt);
                                                    yield Event::ThinkingDelta { delta: tt };
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                _ => {}
                            }
                            if let Some(delta_tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                for tc in delta_tool_calls {
                                    let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                    let entry = tool_calls.entry(index).or_insert_with(|| (String::new(), String::new(), String::new()));
                                    if let Some(id) = tc.get("id").and_then(|v| v.as_str())
                                        && entry.0.is_empty() { entry.0 = id.to_string(); }
                                    if let Some(func) = tc.get("function") {
                                        if let Some(name) = func.get("name").and_then(|v| v.as_str())
                                            && entry.1.is_empty() && !name.is_empty() {
                                                entry.1 = name.to_string();
                                                yield Event::ToolCallStart { id: entry.0.clone(), name: entry.1.clone() };
                                            }
                                        if let Some(args) = func.get("arguments").and_then(|v| v.as_str())
                                            && !args.is_empty() {
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
                            let (stop, err_msg) = crate::simple_options::map_openai_finish_reason(reason);
                            if let Some(msg) = err_msg {
                                partial.error_message = Some(msg);
                            }
                            partial.stop_reason = Some(stop);
                        }
                    }
                }

                if let Some(usage) = chunk.get("usage") {
                    partial.usage = Some(crate::simple_options::parse_openai_usage(usage, model));
                }
            }
            if got_done {
                break;
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

        if !current_thinking.is_empty() && !partial.content.iter().any(|b| matches!(b, ContentBlock::Thinking { .. })) {
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
        for (_idx, (id, name, args_json)) in tool_calls {
            let parsed = crate::jsonparse::parse_streaming_json(&args_json);
            let arguments = match &parsed {
                serde_json::Value::Object(map) => map.clone().into_iter().collect(),
                _ => std::collections::HashMap::new(),
            };
            partial.content.push(ContentBlock::ToolCall {
                id: id.clone(), name: name.clone(), arguments, thought_signature: None,
            });
            yield Event::ToolCallEnd { id, name, arguments: parsed };
        }
        match partial.stop_reason.clone() {
            Some(StopReason::Error) => {
                let msg = partial.error_message.clone().unwrap_or_else(|| "Provider returned an error stop reason".to_string());
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                    message: Some(partial),
                };
            }
            None => {
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                        "Stream ended without finish_reason".to_string(),
                    )),
                    message: Some(partial),
                };
            }
            Some(reason) => {
                yield Event::Done { reason, message: partial };
            }
        }
    })
}

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;

/// Encode a u64 in lowercase base36.
fn to_base36(mut n: u64) -> String {
    const ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return "0".to_string();
    }
    let mut s = Vec::new();
    while n > 0 {
        s.push(ALPHABET[(n % 36) as usize]);
        n /= 36;
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

/// Derive a candidate Mistral tool-call id (mirrors deriveMistralToolCallId).
fn derive_mistral_tool_call_id(id: &str, attempt: u32) -> String {
    let normalized: String = id.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() { id.to_string() } else { normalized };
    let seed = if attempt == 0 { seed_base } else { format!("{seed_base}:{attempt}") };
    to_base36(crate::utils::hash_string(&seed))
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

/// Stateful normalizer mapping arbitrary tool-call ids to unique 9-char
/// alphanumeric Mistral ids (mirrors createMistralToolCallIdNormalizer).
#[derive(Default)]
struct MistralIdNormalizer {
    id_map: std::collections::HashMap<String, String>,
    reverse: std::collections::HashMap<String, String>,
}

impl MistralIdNormalizer {
    fn normalize(&mut self, id: &str) -> String {
        if let Some(existing) = self.id_map.get(id) {
            return existing.clone();
        }
        let mut attempt = 0;
        loop {
            let candidate = derive_mistral_tool_call_id(id, attempt);
            match self.reverse.get(&candidate) {
                Some(owner) if owner != id => attempt += 1,
                _ => {
                    self.id_map.insert(id.to_string(), candidate.clone());
                    self.reverse.insert(candidate.clone(), id.to_string());
                    return candidate;
                }
            }
        }
    }
}

pub(crate) fn build_mistral_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let mut messages = Vec::new();
    let mut id_normalizer = MistralIdNormalizer::default();

    if let Some(ref prompt) = context.system_prompt {
        messages.push(json!({"role": "system", "content": prompt}));
    }

    let transformed_messages = crate::transform::transform_messages(&context.messages, model);
    let supports_images = model.input.iter().any(|i| i == "image");

    for msg in &transformed_messages {
        match msg.role {
            Role::User => {
                let text_only: Vec<&str> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                }).collect();
                if msg.content.len() == 1 && text_only.len() == 1 {
                    messages.push(json!({"role": "user", "content": text_only[0]}));
                } else {
                    let parts: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(json!({"type": "text", "text": text})),
                        ContentBlock::Image { data, mime_type } if supports_images => Some(json!({
                            "type": "image_url", "image_url": format!("data:{};base64,{}", mime_type, data)
                        })),
                        _ => None,
                    }).collect();
                    if !parts.is_empty() {
                        messages.push(json!({"role": "user", "content": parts}));
                    }
                }
            }
            Role::Assistant => {
                let text = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } if !text.trim().is_empty() => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("");
                let tool_calls: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::ToolCall { id, name, arguments, .. } => Some(json!({
                        "id": id_normalizer.normalize(id),
                        "type": "function",
                        "function": {"name": name, "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())}
                    })),
                    _ => None,
                }).collect();
                if text.is_empty() && tool_calls.is_empty() { continue; }
                let mut m = json!({"role": "assistant"});
                if !text.is_empty() {
                    m["content"] = json!(text);
                }
                if !tool_calls.is_empty() {
                    m["tool_calls"] = json!(tool_calls);
                }
                messages.push(m);
            }
            Role::ToolResult => {
                let text_result = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                let has_images = msg.content.iter().any(|b| matches!(b, ContentBlock::Image { .. }));
                let tool_text = build_tool_result_text(&text_result, has_images, supports_images, msg.is_error);
                let mut m = json!({
                    "role": "tool",
                    "content": tool_text,
                });
                if let Some(ref id) = msg.tool_call_id {
                    m["tool_call_id"] = json!(id_normalizer.normalize(id));
                }
                if let Some(ref name) = msg.tool_name {
                    m["name"] = json!(name);
                }
                messages.push(m);
            }
        }
    }

    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true,
    });

    if let Some(max) = opts.max_tokens {
        payload["max_tokens"] = json!(max);
    }
    if let Some(temp) = opts.temperature {
        payload["temperature"] = json!(temp);
    }

    if !context.tools.is_empty() {
        let tools: Vec<Value> = context.tools.iter().map(|t| {
            json!({
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters}
            })
        }).collect();
        payload["tools"] = json!(tools);
    }

    // Tool choice: string variants pass through; objects pass through.
    if let Some(ref tc) = opts.tool_choice {
        payload["tool_choice"] = tc.clone();
    }

    // Reasoning: Mistral uses prompt_mode for most reasoning models and reasoning_effort
    // for a specific set (mirrors upstream usesReasoningEffort / usesPromptModeReasoning).
    if model.reasoning && let Some(level) = opts.reasoning.as_ref() {
        let uses_reasoning_effort = matches!(model.id.as_str(),
            "mistral-small-2603" | "mistral-small-latest" | "mistral-medium-3.5");
        if uses_reasoning_effort {
            let key = format!("{:?}", level).to_lowercase();
            let effort = model.thinking_level_map.as_ref()
                .and_then(|m| m.get(&key))
                .and_then(|v| v.clone())
                .unwrap_or_else(|| "high".to_string());
            payload["reasoning_effort"] = json!(effort);
        } else {
            payload["prompt_mode"] = json!("reasoning");
        }
    }

    payload
}

/// Build the text body for a Mistral tool-result message (mirrors upstream buildToolResultText).
fn build_tool_result_text(text: &str, has_images: bool, supports_images: bool, is_error: bool) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };
    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{}{}{}", error_prefix, trimmed, image_suffix);
    }
    if has_images {
        if supports_images {
            return if is_error { "[tool error] (see attached image)".to_string() } else { "(see attached image)".to_string() };
        }
        return if is_error {
            "[tool error] (image omitted: model does not support images)".to_string()
        } else {
            "(image omitted: model does not support images)".to_string()
        };
    }
    if is_error { "[tool error] (no tool output)".to_string() } else { "(no tool output)".to_string() }
}
