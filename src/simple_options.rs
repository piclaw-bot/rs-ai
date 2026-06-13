//! Thinking level mapping and simple option helpers.

use std::collections::HashMap;
use crate::types::{Model, ThinkingLevel, ModelThinkingLevel};

/// Default thinking budget per level (tokens).
pub fn default_thinking_budgets() -> HashMap<ThinkingLevel, u32> {
    HashMap::from([
        (ThinkingLevel::Minimal, 1024),
        (ThinkingLevel::Low, 2048),
        (ThinkingLevel::Medium, 8192),
        (ThinkingLevel::High, 16384),
    ])
}

/// Extended levels in order.
const LEVELS: &[ModelThinkingLevel] = &[
    ModelThinkingLevel::Off,
    ModelThinkingLevel::Minimal,
    ModelThinkingLevel::Low,
    ModelThinkingLevel::Medium,
    ModelThinkingLevel::High,
    ModelThinkingLevel::XHigh,
];

/// Get supported thinking levels for a model.
pub fn get_supported_thinking_levels(model: &Model) -> Vec<ModelThinkingLevel> {
    if !model.reasoning {
        return vec![ModelThinkingLevel::Off];
    }
    let map = model.thinking_level_map.as_ref();
    let mut out = Vec::new();
    for level in LEVELS {
        let key = level.to_string();
        if let Some(m) = map {
            match m.get(&key) {
                Some(None) => continue, // explicitly disabled
                Some(Some(_)) => out.push(level.clone()),
                None => {
                    // xhigh must be explicit
                    if *level == ModelThinkingLevel::XHigh {
                        continue;
                    }
                    out.push(level.clone());
                }
            }
        } else {
            if *level == ModelThinkingLevel::XHigh {
                continue;
            }
            out.push(level.clone());
        }
    }
    if out.is_empty() {
        vec![ModelThinkingLevel::Off]
    } else {
        out
    }
}

/// Clamp a requested level to the nearest supported level (preferring downgrade).
pub fn clamp_thinking_level(model: &Model, level: &ModelThinkingLevel) -> ModelThinkingLevel {
    let available = get_supported_thinking_levels(model);
    if available.contains(level) {
        return level.clone();
    }
    let idx = LEVELS.iter().position(|l| l == level);
    let idx = match idx {
        Some(i) => i,
        None => return available[0].clone(),
    };
    // Prefer downgrade
    for i in (0..idx).rev() {
        if available.contains(&LEVELS[i]) {
            return LEVELS[i].clone();
        }
    }
    // Then upgrade
    for level in LEVELS.iter().skip(idx + 1) {
        if available.contains(level) {
            return level.clone();
        }
    }
    available[0].clone()
}

/// Map a thinking level to its provider-specific string value.
pub fn map_thinking_level(model: &Model, level: &ModelThinkingLevel) -> Option<String> {
    let clamped = clamp_thinking_level(model, level);
    if let Some(ref map) = model.thinking_level_map
        && let Some(mapped) = map.get(&clamped.to_string()) {
            return mapped.clone();
        }
    if clamped == ModelThinkingLevel::Off {
        return Some("none".to_string());
    }
    Some(clamped.to_string())
}

impl std::fmt::Display for ModelThinkingLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Minimal => write!(f, "minimal"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::XHigh => write!(f, "xhigh"),
        }
    }
}

/// Calculate cost from model pricing and usage.
pub fn calculate_cost(model: &Model, usage: &crate::types::Usage) -> crate::types::CostBreakdown {
    let m = 1_000_000.0;
    let input = f64::from(usage.input) * model.cost.input / m;
    let output = f64::from(usage.output) * model.cost.output / m;
    let cache_read = f64::from(usage.cache_read) * model.cost.cache_read / m;
    let cache_write = f64::from(usage.cache_write) * model.cost.cache_write / m;
    crate::types::CostBreakdown {
        input, output, cache_read, cache_write,
        total: input + output + cache_read + cache_write,
    }
}

/// Map an OpenAI-style `finish_reason` to a stop reason plus optional error message
/// (mirrors upstream `mapStopReason`).
pub fn map_openai_finish_reason(reason: &str) -> (crate::types::StopReason, Option<String>) {
    use crate::types::StopReason;
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (StopReason::Error, Some("Provider finish_reason: content_filter".to_string())),
        "network_error" => (StopReason::Error, Some("Provider finish_reason: network_error".to_string())),
        other => (StopReason::Error, Some(format!("Provider finish_reason: {}", other))),
    }
}

/// Parse OpenAI-style chunk usage, accounting for cache read/write tokens and cost
/// (mirrors upstream `parseChunkUsage`).
pub fn parse_openai_usage(raw: &serde_json::Value, model: &Model) -> crate::types::Usage {
    let prompt_tokens = raw.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let cache_read = raw.pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| raw.get("prompt_cache_hit_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0) as u32;
    let cache_write = raw.pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let input = prompt_tokens.saturating_sub(cache_read).saturating_sub(cache_write);
    let output = raw.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let mut usage = crate::types::Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost: Default::default(),
    };
    usage.cost = calculate_cost(model, &usage);
    usage
}

/// Parse OpenAI Responses-style usage (input_tokens/output_tokens with cached
/// tokens) including cost.
pub fn parse_responses_usage(raw: &serde_json::Value, model: &Model) -> crate::types::Usage {
    let cached = raw.pointer("/input_tokens_details/cached_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let input_total = raw.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let input = input_total.saturating_sub(cached);
    let output = raw.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let total = raw.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or((input + output + cached) as u64) as u32;
    let mut usage = crate::types::Usage {
        input, output, cache_read: cached, cache_write: 0, total_tokens: total, cost: Default::default(),
    };
    usage.cost = calculate_cost(model, &usage);
    usage
}

/// Recompute cost for a usage record (for providers that build usage incrementally
/// across events). The provider-supplied `total_tokens` is preserved.
pub fn finalize_usage(model: &Model, usage: &mut crate::types::Usage) {
    usage.cost = calculate_cost(model, usage);
}

/// Clamp xhigh to high for legacy callers.
pub fn clamp_reasoning(level: &ThinkingLevel) -> ThinkingLevel {
    match level {
        ThinkingLevel::XHigh => ThinkingLevel::High,
        other => other.clone(),
    }
}

/// Clamp a requested reasoning level to what the model actually supports.
///
/// Returns `None` when the level clamps to `off` (reasoning disabled), mirroring
/// upstream's `reasoningEffort = clampedReasoning === "off" ? undefined`.
pub fn clamp_reasoning_for_model(model: &Model, level: &ThinkingLevel) -> Option<ThinkingLevel> {
    let requested = match level {
        ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => ModelThinkingLevel::Low,
        ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        ThinkingLevel::High => ModelThinkingLevel::High,
        ThinkingLevel::XHigh => ModelThinkingLevel::XHigh,
    };
    match clamp_thinking_level(model, &requested) {
        ModelThinkingLevel::Off => None,
        ModelThinkingLevel::Minimal => Some(ThinkingLevel::Minimal),
        ModelThinkingLevel::Low => Some(ThinkingLevel::Low),
        ModelThinkingLevel::Medium => Some(ThinkingLevel::Medium),
        ModelThinkingLevel::High => Some(ThinkingLevel::High),
        ModelThinkingLevel::XHigh => Some(ThinkingLevel::XHigh),
    }
}

/// Check if a model supports xhigh thinking.
pub fn supports_xhigh(model: &Model) -> bool {
    get_supported_thinking_levels(model)
        .contains(&ModelThinkingLevel::XHigh)
}

/// Adjust max tokens for thinking budget.
pub fn adjust_max_tokens_for_thinking(
    base_max_tokens: u32,
    model_max_tokens: u32,
    level: &ThinkingLevel,
    budgets: &std::collections::HashMap<ThinkingLevel, u32>,
) -> (u32, u32) {
    let clamped = clamp_reasoning(level);
    let thinking_budget = budgets.get(&clamped).copied()
        .unwrap_or_else(|| default_thinking_budgets().get(&clamped).copied().unwrap_or(8192));
    
    let mut max_tokens = base_max_tokens + thinking_budget;
    if model_max_tokens > 0 && max_tokens > model_max_tokens {
        max_tokens = model_max_tokens;
    }
    let min_output = 1024u32;
    let available = max_tokens.saturating_sub(min_output);
    let final_budget = thinking_budget.min(available);
    (max_tokens, final_budget)
}
