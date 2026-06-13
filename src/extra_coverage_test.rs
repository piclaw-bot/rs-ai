//! Additional test coverage for validation, session resources, prompt cache, JSON parse.

#[cfg(test)]
mod tests {
    use crate::validation::*;
    use crate::session_resources::*;
    use crate::prompt_cache::*;
    use crate::jsonparse::*;
    use crate::diagnostics::*;
    use crate::types::*;
    use serde_json::json;

    // --- Validation ---

    #[test]
    fn test_validate_empty_context() {
        let ctx = Context { system_prompt: None, messages: vec![], tools: vec![] };
        let errs = validate_context(&ctx).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("at least one message")));
    }

    #[test]
    fn test_validate_valid_context() {
        let ctx = Context {
            system_prompt: Some("sys".into()),
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "t".into(), description: "d".into(), parameters: json!({}) }],
        };
        assert!(validate_context(&ctx).is_ok());
    }

    #[test]
    fn test_validate_tool_no_name() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "".into(), description: "d".into(), parameters: json!({}) }],
        };
        let errs = validate_context(&ctx).unwrap_err();
        assert!(errs[0].message.contains("name"));
    }

    #[test]
    fn test_validate_tool_arguments() {
        let tool = Tool { name: "t".into(), description: "d".into(), parameters: json!({"type": "object"}) };
        assert!(validate_tool_arguments(&tool, &json!({"key": "val"})).is_ok());
        assert!(validate_tool_arguments(&tool, &json!("string")).is_err());
    }

    #[test]
    fn test_validate_tool_call() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "search".into(), description: "s".into(), parameters: json!({}) }],
        };
        assert!(validate_tool_call(&ctx, "search", &json!({})).is_ok());
        assert!(validate_tool_call(&ctx, "unknown", &json!({})).is_err());
    }

    #[test]
    fn test_tool_call_limit() {
        let calls: Vec<serde_json::Value> = (0..200).map(|i| json!({"id": i})).collect();
        let config = default_tool_call_limit_config();
        let limited = apply_tool_call_limit(&calls, &config);
        assert_eq!(limited.len(), 128);
    }

    // --- Session Resources ---

    #[test]
    fn test_session_register_unregister() {
        let sr = SessionResources::new();
        sr.register("s1", "codex");
        sr.register("s2", "codex");
        assert_eq!(sr.list().len(), 2);
        sr.unregister("s1");
        assert_eq!(sr.list().len(), 1);
    }

    #[test]
    fn test_session_cleanup_all() {
        let sr = SessionResources::new();
        sr.register("s1", "codex");
        sr.register("s2", "codex");
        sr.register("s3", "bedrock");
        assert_eq!(cleanup_session_resources(&sr), 3);
        assert_eq!(sr.list().len(), 0);
    }

    #[test]
    fn test_register_session_resource_cleanup() {
        let sr = SessionResources::new();
        register_session_resource_cleanup(&sr, "s1", "codex");
        assert_eq!(sr.list().len(), 1);
    }

    // --- Prompt Cache ---

    #[test]
    fn test_should_cache_large_model() {
        let model = Model {
            context_window: 128000, ..base_model()
        };
        let opts = StreamOptions::default();
        assert!(should_cache(&model, &opts));
    }

    #[test]
    fn test_should_cache_small_model() {
        let model = Model {
            context_window: 4096, ..base_model()
        };
        let opts = StreamOptions::default();
        assert!(!should_cache(&model, &opts));
    }

    #[test]
    fn test_should_cache_explicit_none() {
        let model = Model { context_window: 128000, ..base_model() };
        let opts = StreamOptions { cache_retention: Some(CacheRetention::None), ..Default::default() };
        assert!(!should_cache(&model, &opts));
    }

    #[test]
    fn test_cache_session_id_format() {
        let model = base_model();
        let id = cache_session_id(&model, 0xdeadbeef);
        assert!(id.starts_with("openai:test:"));
        assert!(id.contains("deadbeef"));
    }

    #[test]
    fn test_clamp_prompt_cache_key() {
        let short = "short-key";
        assert_eq!(clamp_openai_prompt_cache_key(short), short);
        let long: String = "x".repeat(100);
        assert_eq!(clamp_openai_prompt_cache_key(&long).len(), 64);
    }

    // --- JSON Parse ---

    #[test]
    fn test_parse_complete_json() {
        let v = parse_partial_json(r#"{"a": 1, "b": [2, 3]}"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["a"], 1);
    }

    #[test]
    fn test_parse_partial_object() {
        let v = parse_partial_json(r#"{"key": "val"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["key"], "val");
    }

    #[test]
    fn test_parse_partial_nested() {
        let v = parse_partial_json(r#"{"a": {"b": [1, 2"#);
        assert!(v.is_some());
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(parse_partial_json("").is_none());
        assert!(parse_partial_json("   ").is_none());
    }

    // --- Diagnostics ---

    #[test]
    fn test_create_diagnostic() {
        let d = create_assistant_message_diagnostic("provider_error", "connection reset");
        assert_eq!(d.diagnostic_type, "provider_error");
        assert_eq!(d.error.message, "connection reset");
        assert!(d.timestamp > 0);
    }

    #[test]
    fn test_extract_diagnostic_error() {
        let d = create_assistant_message_diagnostic("test", "error msg");
        assert_eq!(extract_diagnostic_error(&d), "error msg");
    }

    #[test]
    fn test_transport_failure_diagnostic() {
        let d = transport_failure_diagnostic("ws setup failed");
        assert_eq!(d.diagnostic_type, "provider_transport_failure");
        assert_eq!(d.error.name.as_deref(), Some("TransportError"));
    }

    // --- Helpers ---

    fn base_model() -> Model {
        Model {
            id: "test".into(), name: "Test".into(), api: "openai-completions".into(),
            provider: "openai".into(), base_url: "".into(), reasoning: false,
            thinking_level_map: None, input: vec!["text".into()],
            cost: ModelCost::default(), context_window: 128000, max_tokens: 4096,
            headers: None, api_key: None,
            compat: Default::default(),
        }
    }
}
