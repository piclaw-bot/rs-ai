#[cfg(test)]
mod tests {
    use crate::harness::*;
    use crate::types::*;
    use std::collections::HashMap;

    fn sample_context() -> Context {
        Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![
                user_message("Hello"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text { text: "Hi there!".into(), text_signature: None }],
                    timestamp: 0,
                    api: None, provider: None, model: None, response_id: None,
                    response_model: None, diagnostics: Vec::new(),
                    usage: None, stop_reason: Some(StopReason::Stop), error_message: None,
                    tool_call_id: None, tool_name: None, is_error: false, details: None,
                },
            ],
            tools: vec![],
        }
    }

    fn sample_model() -> Model {
        Model {
            id: "test".into(), name: "Test".into(), api: "test".into(),
            provider: "test".into(), base_url: "".into(), reasoning: false,
            thinking_level_map: None, input: vec!["text".into()],
            cost: ModelCost::default(), context_window: 4096, max_tokens: 1024,
            headers: None, api_key: None,
            compat: Default::default(),
        }
    }

    #[test]
    fn test_get_text_multiblock() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking { thinking: "hmm".into(), thinking_signature: None, redacted: false },
                ContentBlock::Text { text: "Answer: ".into(), text_signature: None },
                ContentBlock::Text { text: "42".into(), text_signature: None },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        };
        assert_eq!(get_text_content(&msg), "Answer: 42");
    }

    #[test]
    fn test_append_and_save_load() {
        let ctx = sample_context();
        let ctx = append_user_message(ctx, "How are you?");
        assert_eq!(ctx.messages.len(), 3);

        let json = save_context(&ctx).unwrap();
        let loaded = load_context(&json).unwrap();
        assert_eq!(loaded.messages.len(), 3);
        assert_eq!(loaded.system_prompt, ctx.system_prompt);
    }

    #[test]
    fn test_tool_result_flow() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall {
                id: "tc1".into(),
                name: "search".into(),
                arguments: HashMap::from([("q".into(), serde_json::json!("rust"))]),
                thought_signature: None,
            }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: Some(StopReason::ToolUse), error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        };
        assert!(is_tool_use(&msg));
        assert!(has_tool_calls(&msg));
        assert!(needs_tool_execution(&msg));

        let calls = get_tool_calls(&msg);
        assert_eq!(calls.len(), 1);

        let ctx = sample_context();
        let ctx = append_tool_result(ctx, "tc1", "search", "found results", false);
        assert_eq!(ctx.messages.last().unwrap().role, Role::ToolResult);
    }

    #[test]
    fn test_fits_in_context() {
        let ctx = sample_context();
        let model = sample_model();
        assert!(fits_in_context_window(&ctx, &model));
    }

    #[test]
    fn test_models_are_equal() {
        let a = sample_model();
        let b = Model { id: "other".into(), ..a.clone() };
        assert!(models_are_equal(&a, &a));
        assert!(!models_are_equal(&a, &b));
    }

    #[test]
    fn test_invoke_on_payload() {
        let payload = serde_json::json!({"model": "test"});
        let hook = |mut p: serde_json::Value| { p["added"] = serde_json::json!(true); p };
        let result = invoke_on_payload(payload, Some(&hook));
        assert_eq!(result["added"], true);
    }
}
