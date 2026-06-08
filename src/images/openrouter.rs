//! OpenRouter image generation provider.

use std::sync::Arc;
use std::time::Duration;
use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use super::types::*;
use crate::env::get_env_api_key;
use crate::types::{ModelCost, StopReason, Usage, CostBreakdown};

/// Options for image generation.
#[derive(Debug, Clone, Default)]
pub struct ImagesOptions {
    pub api_key: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub timeout: Option<Duration>,
    pub max_retries: u32,
    pub max_retry_delay_ms: u64,
}

/// Generate images via OpenRouter.
pub async fn generate_openrouter(
    model: &ImagesModel,
    context: &ImagesContext,
    opts: &ImagesOptions,
) -> AssistantImages {
    let mut out = AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        stop_reason: StopReason::Stop,
        timestamp: chrono_timestamp(),
        response_id: None,
        usage: None,
        error_message: None,
    };

    let api_key = opts.api_key.clone()
        .or_else(|| get_env_api_key(&model.provider));
    let api_key = match api_key {
        Some(k) => k,
        None => {
            out.stop_reason = StopReason::Error;
            out.error_message = Some(format!("no API key for provider: {}", model.provider));
            return out;
        }
    };

    if context.input.is_empty() {
        out.stop_reason = StopReason::Error;
        out.error_message = Some("image context has no inputs".into());
        return out;
    }

    let payload = build_payload(model, context);
    let body = match serde_json::to_vec(&payload) {
        Ok(b) => b,
        Err(e) => {
            out.stop_reason = StopReason::Error;
            out.error_message = Some(e.to_string());
            return out;
        }
    };

    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key)).unwrap());

    if let Some(ref extra) = opts.headers {
        for (k, v) in extra {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                HeaderValue::from_str(v),
            ) {
                headers.insert(name, val);
            }
        }
    }

    let client = reqwest::Client::new();
    let mut last_err = String::new();

    for attempt in 0..=opts.max_retries {
        let resp = client
            .post(&url)
            .headers(headers.clone())
            .body(body.clone())
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = e.to_string();
                if attempt < opts.max_retries {
                    tokio::time::sleep(Duration::from_millis(250 * (attempt as u64 + 1))).await;
                    continue;
                }
                out.stop_reason = StopReason::Error;
                out.error_message = Some(last_err);
                return out;
            }
        };

        let status = resp.status().as_u16();
        if status == 429 || status >= 500 {
            last_err = format!("HTTP {}", status);
            if attempt < opts.max_retries {
                tokio::time::sleep(Duration::from_millis(250 * (attempt as u64 + 1))).await;
                continue;
            }
            out.stop_reason = StopReason::Error;
            out.error_message = Some(last_err);
            return out;
        }

        if status >= 300 {
            let body_text = resp.text().await.unwrap_or_default();
            out.stop_reason = StopReason::Error;
            out.error_message = Some(format!("HTTP {}: {}", status, body_text));
            return out;
        }

        let raw: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                out.stop_reason = StopReason::Error;
                out.error_message = Some(e.to_string());
                return out;
            }
        };

        parse_response(&raw, model, &mut out);
        return out;
    }

    out.stop_reason = StopReason::Error;
    out.error_message = Some(last_err);
    out
}

fn build_payload(model: &ImagesModel, context: &ImagesContext) -> Value {
    let content: Vec<Value> = context.input.iter().map(|input| match input {
        ImageInput::Text { text } => json!({"type": "text", "text": text}),
        ImageInput::Image { data, mime_type } => json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{};base64,{}", mime_type, data)}
        }),
    }).collect();

    let modalities: Vec<&str> = if model.output.iter().any(|o| o == "text") {
        vec!["image", "text"]
    } else {
        vec!["image"]
    };

    json!({
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": false,
        "modalities": modalities,
    })
}

fn parse_response(raw: &Value, model: &ImagesModel, out: &mut AssistantImages) {
    if let Some(id) = raw.get("id").and_then(|v| v.as_str()) {
        out.response_id = Some(id.to_string());
    }

    if let Some(usage) = raw.get("usage") {
        out.usage = Some(parse_usage(usage, model));
    }

    if let Some(choices) = raw.get("choices").and_then(|v| v.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(msg) = choice.get("message") {
                if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        out.output.push(ImageOutput::Text { text: text.to_string() });
                    }
                }
                if let Some(images) = msg.get("images").and_then(|v| v.as_array()) {
                    for img in images {
                        let url = img.get("image_url")
                            .and_then(|v| v.as_object())
                            .and_then(|o| o.get("url"))
                            .and_then(|v| v.as_str())
                            .or_else(|| img.get("image_url").and_then(|v| v.as_str()));
                        if let Some(u) = url {
                            if let Some(rest) = u.strip_prefix("data:") {
                                if let Some((mime, data)) = rest.split_once(";base64,") {
                                    out.output.push(ImageOutput::Image {
                                        data: data.to_string(),
                                        mime_type: mime.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn parse_usage(raw: &Value, model: &ImagesModel) -> Usage {
    let prompt = raw.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let completion = raw.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let cached = raw.pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let cache_write = raw.pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let cache_read = if cache_write > 0 { cached.saturating_sub(cache_write) } else { cached };
    let input = prompt.saturating_sub(cache_read + cache_write);

    let m = 1_000_000.0;
    Usage {
        input,
        output: completion,
        cache_read,
        cache_write,
        total_tokens: input + completion + cache_read + cache_write,
        cost: CostBreakdown {
            input: f64::from(input) * model.cost.input / m,
            output: f64::from(completion) * model.cost.output / m,
            cache_read: f64::from(cache_read) * model.cost.cache_read / m,
            cache_write: f64::from(cache_write) * model.cost.cache_write / m,
            total: 0.0, // computed below
        },
    }
}

fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
