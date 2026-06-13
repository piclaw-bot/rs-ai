//! OpenAI Responses API provider (also serves Azure OpenAI Responses).

use std::sync::Arc;

use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::compat::detect_compat;
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
    stream_responses_inner(model, context, opts, false)
}

/// Start an Azure OpenAI Responses stream (api-key auth, api-version, session headers,
/// and Azure reasoning-event normalization).
pub fn stream_azure_responses<'a>(
    model: &'a Model,
    context: &'a Context,
    opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    stream_responses_inner(model, context, opts, true)
}

fn stream_responses_inner<'a>(
    model: &'a Model,
    context: &'a Context,
    opts: &'a StreamOptions,
    is_azure: bool,
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
    let base = model.base_url.trim_end_matches('/');
    let url = if is_azure {
        let api_version = std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| "v1".to_string());
        format!("{}/responses?api-version={}", base, api_version)
    } else {
        format!("{}/responses", base)
    };

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert("accept", HeaderValue::from_static("text/event-stream"));
    if is_azure {
        if let Ok(val) = HeaderValue::from_str(&api_key) {
            headers.insert("api-key", val);
        }
        // Azure session affinity headers.
        if let Some(ref session_id) = opts.session_id {
            for (k, v) in crate::azure::azure_session_headers(session_id) {
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                    HeaderValue::from_str(&v),
                ) {
                    headers.insert(name, val);
                }
            }
        }
    } else {
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());
        // Session headers (non-Azure): upstream sends both `session_id`
        // (gated on compat.sendSessionIdHeader, default true) and `x-client-request-id`.
        if let Some(ref session_id) = opts.session_id
            && let Ok(val) = HeaderValue::from_str(session_id) {
                headers.insert("session_id", val.clone());
                headers.insert("x-client-request-id", val);
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

                let mut data: Value = match serde_json::from_str(&evt.data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if is_azure {
                    crate::azure::normalize_azure_reasoning_event(&mut data);
                }

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
                    "response.output_text.delta" | "response.refusal.delta" => {
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
                    "response.reasoning_summary_part.done" => {
                        // Separate consecutive summary parts with a blank line (only when a
                        // summary is in progress), matching upstream.
                        if !current_thinking.is_empty() {
                            current_thinking.push_str("\n\n");
                            yield Event::ThinkingDelta { delta: "\n\n".to_string() };
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
                                Some("message") => {
                                    // Capture the message item id/phase as a text signature so
                                    // the assistant text block pairs correctly on replay.
                                    let text = item.get("content").and_then(|v| v.as_array())
                                        .map(|parts| parts.iter().filter_map(|p| {
                                            p.get("text").and_then(|v| v.as_str())
                                                .or_else(|| p.get("refusal").and_then(|v| v.as_str()))
                                        }).collect::<Vec<_>>().join(""))
                                        .filter(|s| !s.is_empty())
                                        .unwrap_or_else(|| std::mem::take(&mut current_text));
                                    let sig = item.get("id").and_then(|v| v.as_str()).map(|id| {
                                        encode_text_signature_v1(id, item.get("phase").and_then(|v| v.as_str()))
                                    });
                                    partial.content.push(ContentBlock::Text {
                                        text,
                                        text_signature: sig,
                                    });
                                    current_text.clear();
                                }
                                _ => {}
                            }
                        }
                    }
                    "response.completed" | "response.incomplete" => {
                        if let Some(response) = data.get("response") {
                            if let Some(model_name) = response.get("model").and_then(|v| v.as_str()) {
                                partial.response_model = Some(model_name.to_string());
                            }
                            if let Some(usage) = response.get("usage") {
                                let mut parsed = crate::simple_options::parse_responses_usage(usage, model);
                                // Resolve the effective service tier (response value wins) and
                                // apply its cost multiplier, matching applyServiceTierPricing.
                                let tier = response.get("service_tier").and_then(|v| v.as_str())
                                    .or(opts.service_tier.as_deref());
                                crate::simple_options::apply_service_tier_pricing(model, &mut parsed, tier);
                                partial.usage = Some(parsed);
                            }
                            // Map response.status, then upgrade to tool-use when tool calls are present.
                            let status = response.get("status").and_then(|v| v.as_str()).unwrap_or("completed");
                            let mut reason = match status {
                                "completed" | "in_progress" | "queued" => StopReason::Stop,
                                "incomplete" => StopReason::Length,
                                "failed" | "cancelled" => StopReason::Error,
                                _ => StopReason::Stop,
                            };
                            if reason == StopReason::Error {
                                let detail = response.pointer("/incomplete_details/reason").and_then(|v| v.as_str())
                                    .or_else(|| response.pointer("/error/message").and_then(|v| v.as_str()))
                                    .unwrap_or(status);
                                partial.error_message = Some(format!("response {}: {}", status, detail));
                            } else if status == "incomplete"
                                && let Some(d) = response.pointer("/incomplete_details/reason").and_then(|v| v.as_str()) {
                                    partial.error_message = Some(format!("incomplete: {}", d));
                            }
                            if reason != StopReason::Error
                                && partial.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. })) {
                                reason = StopReason::ToolUse;
                            }
                            partial.stop_reason = Some(reason);
                        }
                    }
                    "response.failed" => {
                        // error.code: message, else incomplete: reason, else generic (mirrors upstream).
                        let resp = data.get("response");
                        let msg = if let Some(err) = resp.and_then(|r| r.get("error")).filter(|e| !e.is_null()) {
                            let code = err.get("code").and_then(|v| v.as_str()).unwrap_or("unknown");
                            let m = err.get("message").and_then(|v| v.as_str()).unwrap_or("no message");
                            format!("{code}: {m}")
                        } else if let Some(reason) = resp.and_then(|r| r.pointer("/incomplete_details/reason")).and_then(|v| v.as_str()) {
                            format!("incomplete: {reason}")
                        } else {
                            "Unknown error (no error details in response)".to_string()
                        };
                        partial.stop_reason = Some(StopReason::Error);
                        partial.error_message = Some(msg.clone());
                        yield Event::Error {
                            reason: StopReason::Error,
                            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                            message: Some(partial.clone()),
                        };
                        return;
                    }
                    "error" => {
                        let msg = data.pointer("/message").and_then(|v| v.as_str())
                            .or_else(|| data.pointer("/error/message").and_then(|v| v.as_str()))
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "Responses stream error".to_string());
                        let code = data.get("code").and_then(|v| v.as_str()).map(|c| format!("Error Code {}: ", c)).unwrap_or_default();
                        partial.stop_reason = Some(StopReason::Error);
                        partial.error_message = Some(format!("{}{}", code, msg));
                        yield Event::Error {
                            reason: StopReason::Error,
                            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(format!("{}{}", code, msg))),
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

        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text {
                text: current_text,
                text_signature: None,
            });
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
            Some(reason) => {
                yield Event::Done { reason, message: partial };
            }
            None => {
                yield Event::Done { reason: StopReason::Stop, message: partial };
            }
        }
    })
}

/// Encode an assistant text item id (+ optional phase) as a v1 text signature
/// (mirrors encodeTextSignatureV1).
fn encode_text_signature_v1(id: &str, phase: Option<&str>) -> String {
    match phase {
        Some(p) => json!({"v": 1, "id": id, "phase": p}).to_string(),
        None => json!({"v": 1, "id": id}).to_string(),
    }
}

/// Parsed assistant text signature: an item id and an optional phase.
struct ParsedTextSignature {
    id: Option<String>,
    phase: Option<String>,
}

/// Providers whose stored tool-call ids use the Responses `callId|itemId` form.
fn is_responses_tool_call_provider(provider: &str) -> bool {
    matches!(provider, "openai" | "openai-codex" | "opencode")
}

/// Sanitize an id part for the Responses API: keep [A-Za-z0-9_-], cap at 64, trim
/// trailing underscores (mirrors normalizeIdPart).
fn normalize_id_part(part: &str) -> String {
    let sanitized: String = part
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let truncated = if sanitized.len() > 64 { sanitized[..64].to_string() } else { sanitized };
    truncated.trim_end_matches('_').to_string()
}

/// Build a foreign-history item id (mirrors buildForeignResponsesItemId).
fn build_foreign_responses_item_id(item_id: &str) -> String {
    let s = format!("fc_{}", crate::utils::short_hash(item_id));
    if s.len() > 64 { s[..64].to_string() } else { s }
}

/// Resolve the `call_id` and optional item `id` for a Responses `function_call`
/// item from a stored tool-call id, applying upstream normalizeToolCallId plus the
/// isDifferentModel item-id omission rule.
fn responses_function_call_ids(
    raw_id: &str,
    model: &Model,
    src_provider: Option<&str>,
    src_api: Option<&str>,
    src_model: Option<&str>,
) -> (String, Option<String>) {
    if !is_responses_tool_call_provider(&model.provider) || !raw_id.contains('|') {
        return (normalize_id_part(raw_id), None);
    }
    let (call_part, item_part) = raw_id.split_once('|').unwrap();
    let call_id = normalize_id_part(call_part);
    let is_foreign = src_provider != Some(model.provider.as_str()) || src_api != Some(model.api.as_str());
    let mut item_id = if is_foreign {
        build_foreign_responses_item_id(item_part)
    } else {
        normalize_id_part(item_part)
    };
    if !item_id.starts_with("fc_") {
        item_id = normalize_id_part(&format!("fc_{item_id}"));
    }
    // For a different model on the same provider/api, omit the item id to avoid
    // OpenAI's fc/rs pairing validation.
    let is_different_model = src_model != Some(model.id.as_str())
        && src_provider == Some(model.provider.as_str())
        && src_api == Some(model.api.as_str());
    if is_different_model && item_id.starts_with("fc_") {
        return (call_id, None);
    }
    (call_id, Some(item_id))
}

/// Resolve the `call_id` for a Responses `function_call_output` from a stored
/// tool-call id (normalizeToolCallId then split on `|`).
fn responses_function_output_call_id(raw_id: &str, model: &Model) -> String {
    if !is_responses_tool_call_provider(&model.provider) || !raw_id.contains('|') {
        return normalize_id_part(raw_id);
    }
    let (call_part, _) = raw_id.split_once('|').unwrap();
    normalize_id_part(call_part)
}

/// Parse a text signature into an item id (and optional phase), mirroring upstream
/// parseTextSignature: JSON `{v:1,id,phase}` form, else legacy plain-string id.
fn parse_text_signature(signature: Option<&str>) -> Option<ParsedTextSignature> {
    let sig = signature?;
    if sig.starts_with('{')
        && let Ok(parsed) = serde_json::from_str::<Value>(sig)
        && parsed.get("v").and_then(|v| v.as_i64()) == Some(1)
        && let Some(id) = parsed.get("id").and_then(|v| v.as_str()) {
        let phase = match parsed.get("phase").and_then(|p| p.as_str()) {
            Some(p @ ("commentary" | "final_answer")) => Some(p.to_string()),
            _ => None,
        };
        return Some(ParsedTextSignature { id: Some(id.to_string()), phase });
    }
    Some(ParsedTextSignature { id: Some(sig.to_string()), phase: None })
}

pub(crate) fn build_responses_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    let compat = detect_compat(model);
    let mut input = Vec::new();

    if let Some(ref prompt) = context.system_prompt {
        // Reasoning models use the developer role (matching upstream).
        let role = if model.reasoning && compat.supports_developer_role != Some(false) {
            "developer"
        } else {
            "system"
        };
        input.push(json!({"role": role, "content": prompt}));
    }

    let transformed_messages = crate::transform::transform_messages(&context.messages, model);

    for (msg_index, msg) in transformed_messages.iter().enumerate() {
        match msg.role {
            Role::User => {
                if msg.content.len() == 1 {
                    match &msg.content[0] {
                        ContentBlock::Text { text, .. } => input.push(json!({"role": "user", "content": text})),
                        ContentBlock::Image { data, mime_type } => input.push(json!({
                            "role": "user",
                            "content": [{"type": "input_image", "detail": "auto", "image_url": format!("data:{};base64,{}", mime_type, data)}]
                        })),
                        _ => {}
                    }
                } else {
                    let parts: Vec<Value> = msg.content.iter().filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(json!({"type": "input_text", "text": text})),
                        ContentBlock::Image { data, mime_type } => Some(json!({
                            "type": "input_image", "detail": "auto", "image_url": format!("data:{};base64,{}", mime_type, data)
                        })),
                        _ => None,
                    }).collect();
                    input.push(json!({"role": "user", "content": parts}));
                }
            }
            Role::Assistant => {
                // Emit blocks in content order so encrypted reasoning items pair with the
                // following message/function_call items (matching upstream).
                let mut text_block_index = 0usize;
                for block in &msg.content {
                    match block {
                        ContentBlock::Thinking { thinking_signature: Some(sig), .. } => {
                            if let Ok(v) = serde_json::from_str::<Value>(sig) {
                                input.push(v);
                            }
                        }
                        ContentBlock::Text { text, text_signature } if !text.trim().is_empty() => {
                            // Resolve the assistant message item id/phase from the text
                            // signature, falling back to a deterministic msg_pi_ id
                            // (mirrors upstream parseTextSignature + fallback).
                            let parsed = parse_text_signature(text_signature.as_deref());
                            let fallback = if text_block_index == 0 {
                                format!("msg_pi_{msg_index}")
                            } else {
                                format!("msg_pi_{msg_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let msg_id = match parsed.as_ref().and_then(|p| p.id.clone()) {
                                Some(id) if id.len() > 64 => format!("msg_{}", crate::utils::short_hash(&id)),
                                Some(id) => id,
                                None => fallback,
                            };
                            let mut item = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text, "annotations": []}],
                                "status": "completed",
                                "id": msg_id,
                            });
                            if let Some(phase) = parsed.and_then(|p| p.phase) {
                                item["phase"] = json!(phase);
                            }
                            input.push(item);
                        }
                        ContentBlock::ToolCall { id, name, arguments, .. } => {
                            let (call_id, item_id) = responses_function_call_ids(
                                id,
                                model,
                                msg.provider.as_deref(),
                                msg.api.as_deref(),
                                msg.model.as_deref(),
                            );
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
                let call_id = msg.tool_call_id.as_deref().map(|id| responses_function_output_call_id(id, model)).unwrap_or_default();
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

    let mut payload = json!({
        "model": model.id,
        "input": input,
        "stream": true,
        "store": false,
    });

    // Prompt caching: session id is sent via headers, not the body. The cache key
    // is derived from the (resolved) retention.
    let retention = crate::prompt_cache::resolve_cache_retention(opts.cache_retention.as_ref());
    match retention {
        CacheRetention::None => {}
        CacheRetention::Short => {
            if let Some(ref session_id) = opts.session_id {
                payload["prompt_cache_key"] = json!(crate::prompt_cache::clamp_openai_prompt_cache_key(session_id));
            }
        }
        CacheRetention::Long => {
            if let Some(ref session_id) = opts.session_id {
                payload["prompt_cache_key"] = json!(crate::prompt_cache::clamp_openai_prompt_cache_key(session_id));
            }
            if compat.supports_long_cache_retention != Some(false) {
                payload["prompt_cache_retention"] = json!("24h");
            }
        }
    }

    if let Some(max) = opts.max_tokens {
        payload["max_output_tokens"] = json!(max);
    }
    if let Some(temp) = opts.temperature {
        payload["temperature"] = json!(temp);
    }
    if let Some(ref service_tier) = opts.service_tier {
        payload["service_tier"] = json!(service_tier);
    }

    if let Some(level) = opts.reasoning.as_ref().and_then(|l| crate::simple_options::clamp_reasoning_for_model(model, l)) {
        let key = format!("{:?}", level).to_lowercase();
        let effort = model.thinking_level_map.as_ref()
            .and_then(|m| m.get(&key))
            .and_then(|v| v.clone())
            .unwrap_or(key);
        payload["reasoning"] = json!({
            "effort": effort,
            "summary": opts.reasoning_summary.clone().unwrap_or_else(|| "auto".to_string()),
        });
        payload["include"] = json!(["reasoning.encrypted_content"]);
    } else if opts.reasoning_summary.is_some() {
        // Summary requested without an explicit effort: default to medium (mirrors upstream).
        payload["reasoning"] = json!({
            "effort": "medium",
            "summary": opts.reasoning_summary.clone().unwrap_or_else(|| "auto".to_string()),
        });
        payload["include"] = json!(["reasoning.encrypted_content"]);
    } else if model.reasoning && model.provider != "github-copilot" {
        // Reasoning-capable model with no thinking requested: explicitly disable reasoning
        // unless the model maps `off` to null (mirrors the upstream else-if branch).
        match model.thinking_level_map.as_ref().and_then(|m| m.get("off")) {
            Some(None) => {} // off mapped to null -> omit reasoning entirely
            Some(Some(off)) => { payload["reasoning"] = json!({ "effort": off }); }
            None => { payload["reasoning"] = json!({ "effort": "none" }); }
        }
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
