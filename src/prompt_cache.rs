//! Prompt cache helpers.
//!
//! Utilities for managing prompt caching markers and session affinity
//! for providers that support server-side prompt caching.

use crate::types::{CacheRetention, Model, StreamOptions};

/// Determine if a model/request should use prompt caching.
pub fn should_cache(model: &Model, opts: &StreamOptions) -> bool {
    match opts.cache_retention {
        Some(CacheRetention::None) => false,
        Some(CacheRetention::Short) | Some(CacheRetention::Long) => true,
        None => {
            // Default: cache if model supports it (most modern providers do)
            model.context_window >= 32000
        }
    }
}

/// Generate a cache session ID from model and context fingerprint.
pub fn cache_session_id(model: &Model, fingerprint: u64) -> String {
    format!("{}:{}:{:x}", model.provider, model.id, fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, StreamOptions};

    fn test_model(ctx_window: u32) -> Model {
        Model {
            id: "test".into(),
            name: "Test".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: ctx_window,
            max_tokens: 4096,
            headers: None,
            api_key: None,
        }
    }

    #[test]
    fn test_should_cache_explicit() {
        let model = test_model(128000);
        let opts = StreamOptions { cache_retention: Some(CacheRetention::Long), ..Default::default() };
        assert!(should_cache(&model, &opts));
        let opts_none = StreamOptions { cache_retention: Some(CacheRetention::None), ..Default::default() };
        assert!(!should_cache(&model, &opts_none));
    }

    #[test]
    fn test_should_cache_default() {
        let big = test_model(128000);
        let small = test_model(4096);
        let opts = StreamOptions::default();
        assert!(should_cache(&big, &opts));
        assert!(!should_cache(&small, &opts));
    }

    #[test]
    fn test_cache_session_id() {
        let model = test_model(128000);
        let id = cache_session_id(&model, 0xdeadbeef);
        assert!(id.starts_with("openai:test:"));
        assert!(id.contains("deadbeef"));
    }
}
