//! Google Gemini CLI (Cloud Code Assist) provider.
//!
//! Uses the same Google Generative AI SSE protocol but with different
//! authentication (OAuth token via Gemini CLI flow).

use std::sync::Arc;

use futures::stream;
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

    let payload = build_geminicli_payload(model, context, opts);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        model.base_url.trim_end_matches('/'),
        url_encode(&model.id),
    );

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

            let chunk: Value = match serde_json::from_str(&evt.data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(candidates) = chunk.get("candidates").and_then(|v| v.as_array()) {
                for candidate in candidates {
                    if let Some(parts) = candidate.pointer("/content/parts").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                if !text.is_empty() {
                                    if !text_started {
                                        text_started = true;
                                        yield Event::TextStart;
                                    }
                                    current_text.push_str(text);
                                    yield Event::TextDelta { delta: text.to_string() };
                                }
                            }
                        }
                    }
                    if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str()) {
                        if text_started {
                            yield Event::TextEnd;
                        }
                        partial.stop_reason = Some(match reason {
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
                    total_tokens: usage.get("totalTokenCount").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                    ..Default::default()
                });
            }
        }

        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text { text: current_text, text_signature: None });
        }
        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}

fn build_geminicli_payload(model: &Model, context: &Context, opts: &StreamOptions) -> Value {
    // Same payload format as Google Generative AI
    crate::provider::google::build_google_payload_public(model, context, opts)
}

fn url_encode(s: &str) -> String {
    s.replace('%', "%25").replace(' ', "%20").replace('+', "%2B").replace('/', "%2F")
}
