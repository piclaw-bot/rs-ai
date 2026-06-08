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
    for i in (idx + 1)..LEVELS.len() {
        if available.contains(&LEVELS[i]) {
            return LEVELS[i].clone();
        }
    }
    available[0].clone()
}

/// Map a thinking level to its provider-specific string value.
pub fn map_thinking_level(model: &Model, level: &ModelThinkingLevel) -> Option<String> {
    let clamped = clamp_thinking_level(model, level);
    if let Some(ref map) = model.thinking_level_map {
        if let Some(mapped) = map.get(&clamped.to_string()) {
            return mapped.clone();
        }
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
