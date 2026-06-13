//! Additional coverage tests matching Go test categories.

#[cfg(test)]
mod tests {
    use crate::retry::*;
    use crate::context::*;
    use crate::simple_options::*;
    use crate::compaction::*;
    use crate::azure::*;
    use crate::logger::*;
    use crate::utils::*;
    use crate::types::*;
    use std::time::Duration;
    use serde_json::json;

    // --- Retry tests ---

    #[test]
    fn test_default_retry_config() {
        let cfg = default_retry_config();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.backoff_multiplier, 2.0);
        assert_eq!(cfg.max_retry_delay_ms, 60_000);
    }

    #[test]
    fn test_no_retry_config() {
        let cfg = no_retry_config();
        assert_eq!(cfg.max_retries, 0);
    }

    #[test]
    fn test_backoff_increases() {
        let cfg = default_retry_config();
        let d0 = compute_backoff(0, &cfg);
        let d1 = compute_backoff(1, &cfg);
        let d2 = compute_backoff(2, &cfg);
        assert!(d1 > d0);
        assert!(d2 > d1);
    }

    #[test]
    fn test_backoff_caps_at_max() {
        let cfg = RetryConfig { max_delay: Duration::from_secs(2), ..default_retry_config() };
        let d = compute_backoff(10, &cfg);
        assert!(d.as_secs_f64() <= 2.0);
    }

    #[test]
    fn test_retryable_status_codes() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(is_retryable_status(502));
        assert!(is_retryable_status(503));
        assert!(is_retryable_status(504));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
        assert!(!is_retryable_status(401));
        assert!(!is_retryable_status(404));
    }

    #[test]
    fn test_parse_retry_after_seconds() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("1"), Some(Duration::from_secs(1)));
        assert_eq!(parse_retry_after("invalid"), None);
        assert_eq!(parse_retry_after(""), None);
    }

    // --- Cost/Token tests ---

    #[test]
    fn test_calculate_cost_basic() {
        let model = Model {
            cost: ModelCost { input: 3.0, output: 15.0, cache_read: 0.3, cache_write: 3.75 },
            ..test_model_base()
        };
        let usage = Usage { input: 1000, output: 500, cache_read: 200, cache_write: 50, total_tokens: 1750, ..Default::default() };
        let cost = calculate_cost(&model, &usage);
        assert!((cost.input - 0.003).abs() < 0.0001);
        assert!((cost.output - 0.0075).abs() < 0.0001);
        assert!((cost.cache_read - 0.00006).abs() < 0.00001);
        assert!(cost.total > 0.0);
    }

    #[test]
    fn test_estimate_tokens_basic() {
        let ctx = Context {
            system_prompt: Some("System prompt with some text.".into()),
            messages: vec![user_message("Hello, how are you doing today?")],
            tools: vec![],
        };
        let tokens = estimate_tokens(&ctx);
        assert!(tokens > 10);
        assert!(tokens < 100);
    }

    // --- Context overflow tests ---

    #[test]
    fn test_overflow_stop_reason_length() {
        let model = Model { context_window: 100, ..test_model_base() };
        // Length stop with output==0 and input filling the window is overflow.
        let msg = Message {
            stop_reason: Some(StopReason::Length),
            usage: Some(Usage { input: 100, output: 0, ..Default::default() }),
            ..base_msg()
        };
        assert!(is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_overflow_error_message() {
        let model = test_model_base();
        let msg = Message {
            stop_reason: Some(StopReason::Error),
            error_message: Some("context_length_exceeded: reduce your input".into()),
            ..base_msg()
        };
        assert!(is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_overflow_token_limit() {
        let model = Model { context_window: 100, ..test_model_base() };
        // Silent overflow: successful stop but input exceeds the context window.
        let msg = Message {
            stop_reason: Some(StopReason::Stop),
            usage: Some(Usage { input: 110, output: 0, total_tokens: 110, ..Default::default() }),
            ..base_msg()
        };
        assert!(is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_no_overflow_normal() {
        let model = test_model_base();
        let msg = Message {
            stop_reason: Some(StopReason::Stop),
            ..base_msg()
        };
        assert!(!is_context_overflow(&msg, &model));
    }

    // --- Azure tests ---

    #[test]
    fn test_azure_normalize_reasoning() {
        let mut event = json!({
            "type": "response.output_item.done",
            "item": { "type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}] }
        });
        normalize_azure_reasoning_event(&mut event);
        assert!(event.pointer("/item/summary").is_some());
        assert!(event.pointer("/item/content").is_none());
    }

    #[test]
    fn test_azure_no_op_for_non_reasoning() {
        let mut event = json!({
            "type": "response.output_item.done",
            "item": { "type": "message", "content": [{"type": "text", "text": "hello"}] }
        });
        normalize_azure_reasoning_event(&mut event);
        assert!(event.pointer("/item/content").is_some()); // unchanged
    }

    #[test]
    fn test_azure_session_headers_populated() {
        let h = azure_session_headers("sess-abc");
        assert_eq!(h.len(), 3);
        assert_eq!(h["session_id"], "sess-abc");
        assert_eq!(h["x-client-request-id"], "sess-abc");
    }

    #[test]
    fn test_azure_session_headers_empty() {
        let h = azure_session_headers("");
        assert!(h.is_empty());
    }

    #[test]
    fn test_strip_azure_tool_call_fields() {
        let mut calls = vec![json!({"id": "tc1", "content_filter_results": {"safe": true}})];
        strip_azure_tool_call_fields(&mut calls);
        assert!(calls[0].get("content_filter_results").is_none());
        assert!(calls[0].get("id").is_some());
    }

    // --- Logger tests ---

    #[test]
    fn test_stderr_logger_does_not_panic() {
        let logger = new_stderr_logger(LogLevel::Warn);
        logger.log(LogLevel::Debug, "should not print", &[]);
        logger.log(LogLevel::Warn, "should print", &[("key", "val")]);
        logger.log(LogLevel::Error, "error", &[]);
    }

    #[test]
    fn test_get_logger_noop_default() {
        // Default logger is a no-op; just verify it doesn't panic
        log_debug("test", &[]);
        log_info("test", &[]);
        log_warn("test", &[]);
        log_error("test", &[]);
    }

    // --- Utils tests ---

    #[test]
    fn test_hash_deterministic() {
        assert_eq!(hash_string("hello"), hash_string("hello"));
        assert_ne!(hash_string("hello"), hash_string("world"));
    }

    #[test]
    fn test_short_hash() {
        let h = short_hash("test");
        assert_eq!(h.len(), 8);
        assert_eq!(short_hash("test"), short_hash("test"));
    }

    #[test]
    fn test_is_cloudflare_provider() {
        assert!(is_cloudflare_provider("cloudflare-workers-ai"));
        assert!(is_cloudflare_provider("cloudflare-ai-gateway"));
        assert!(!is_cloudflare_provider("openai"));
    }

    #[test]
    fn test_resolve_cloudflare_base_url_substitutes_env() {
        unsafe { std::env::set_var("RS_AI_TEST_ACCT", "acct123"); }
        let resolved = resolve_cloudflare_base_url("https://gateway.ai.cloudflare.com/v1/{RS_AI_TEST_ACCT}/openai");
        assert_eq!(resolved, "https://gateway.ai.cloudflare.com/v1/acct123/openai");
        // No placeholders -> pass-through.
        assert_eq!(resolve_cloudflare_base_url("https://example.com"), "https://example.com");
        unsafe { std::env::remove_var("RS_AI_TEST_ACCT"); }
    }

    #[test]
    fn test_clamp_prompt_cache_key_char_safe() {
        let long = "é".repeat(100); // multi-byte chars; must not panic
        let clamped = crate::prompt_cache::clamp_openai_prompt_cache_key(&long);
        assert_eq!(clamped.chars().count(), 64);
    }

    #[test]
    fn test_copilot_headers_structure() {
        let h = copilot_headers();
        assert!(h.contains_key("User-Agent"));
        assert!(h.contains_key("Copilot-Integration-Id"));
        let h2 = copilot_headers_with_intent("completion");
        assert_eq!(h2["openai-intent"], "completion");
    }

    #[test]
    fn test_sanitize_surrogates_noop_for_valid_utf8() {
        assert_eq!(sanitize_surrogates("Hello 🙈"), "Hello 🙈");
        assert_eq!(sanitize_surrogates("plain"), "plain");
        assert_eq!(sanitize_surrogates(""), "");
    }

    // --- Max tokens / thinking adjustment ---

    #[test]
    fn test_adjust_max_tokens_basic() {
        let budgets = default_thinking_budgets();
        let (max, budget) = adjust_max_tokens_for_thinking(Some(4096), 32000, &ThinkingLevel::High, &budgets);
        assert!(max <= 32000);
        assert!(budget > 0);
        assert!(budget <= 16384); // high budget capped at 16384
    }

    #[test]
    fn test_adjust_max_tokens_capped() {
        let budgets = default_thinking_budgets();
        let (max, budget) = adjust_max_tokens_for_thinking(Some(4096), 8000, &ThinkingLevel::High, &budgets);
        assert!(max <= 8000);
        assert!(budget < 16384); // capped by model limit
    }

    // --- Helpers ---

    fn test_model_base() -> Model {
        Model {
            id: "test".into(), name: "Test".into(), api: "openai-completions".into(),
            provider: "openai".into(), base_url: "".into(), reasoning: false,
            thinking_level_map: None, input: vec!["text".into()],
            cost: ModelCost::default(), context_window: 128000, max_tokens: 4096,
            headers: None, api_key: None,
            compat: Default::default(),
        }
    }

    fn base_msg() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "hi".into(), text_signature: None }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }
    }
}
