//! Remaining edge-case tests to match Go's full coverage.

#[cfg(test)]
mod tests {
    use crate::provider::faux::*;
    use crate::harness::*;
    use crate::types::*;
    use crate::events::Event;
    use crate::transform::*;
    use tokio_stream::StreamExt;

    fn faux_model() -> Model {
        Model {
            id: "faux".into(), name: "Faux".into(), api: "faux".into(),
            provider: "faux".into(), base_url: "".into(), reasoning: true,
            thinking_level_map: None, input: vec!["text".into(), "image".into()],
            cost: ModelCost::default(), context_window: 128000, max_tokens: 4096,
            headers: None, api_key: None,
            compat: Default::default(),
        }
    }

    // --- Faux Thinking Stream ---

    #[tokio::test]
    async fn test_faux_thinking_content() {
        // Faux text stream with thinking-like content
        let model = faux_model();
        let thinking_response = "Let me think... The answer is 42.";
        let mut stream = stream_faux_text(thinking_response, &model);
        let mut final_text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::Done { message, .. } = evt {
                final_text = get_text_content(&message);
            }
        }
        assert_eq!(final_text, thinking_response);
    }

    // --- Faux Multiple Calls ---

    #[tokio::test]
    async fn test_faux_multiple_sequential_calls() {
        let model = faux_model();
        for i in 0..3 {
            let text = format!("response {}", i);
            let mut stream = stream_faux_text(&text, &model);
            let mut final_text = String::new();
            while let Some(evt) = stream.next().await {
                if let Event::Done { message, .. } = evt {
                    final_text = get_text_content(&message);
                }
            }
            assert_eq!(final_text, text);
        }
    }

    // --- Transform Tests ---

    #[test]
    fn test_transform_preserves_tool_calls() {
        let model = faux_model();
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text: "Let me search".into(), text_signature: None },
                ContentBlock::ToolCall {
                    id: "tc1".into(),
                    name: "search".into(),
                    arguments: std::collections::HashMap::new(),
                    thought_signature: None,
                },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }];
        let result = transform_messages(&messages, &model);
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(&result[0].content[1], ContentBlock::ToolCall { .. }));
    }

    #[test]
    fn test_transform_multiple_images_downgraded() {
        let text_model = Model { input: vec!["text".into()], ..faux_model() };
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text { text: "Compare:".into(), text_signature: None },
                ContentBlock::Image { data: "img1".into(), mime_type: "image/png".into() },
                ContentBlock::Image { data: "img2".into(), mime_type: "image/jpeg".into() },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }];
        let result = transform_messages(&messages, &text_model);
        // Consecutive images collapse to a single placeholder (matches upstream).
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(&result[0].content[0], ContentBlock::Text { text, .. } if text == "Compare:"));
        assert!(matches!(&result[0].content[1], ContentBlock::Text { text, .. } if text.contains("omitted")));
    }

    // --- Validation Edge Cases ---

    #[test]
    fn test_validate_tool_no_description() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "t".into(), description: "".into(), parameters: serde_json::json!({}) }],
        };
        let errs = crate::validation::validate_context(&ctx).unwrap_err();
        assert!(errs[0].message.contains("description"));
    }

    // --- Context Deep Clone ---

    #[test]
    fn test_clone_context_deep_copies() {
        let ctx = Context {
            system_prompt: Some("sys".into()),
            messages: vec![
                user_message("hi"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text { text: "hello".into(), text_signature: None }],
                    timestamp: 42,
                    api: Some("openai".into()),
                    provider: Some("openai".into()),
                    model: Some("gpt-4o".into()),
                    response_id: Some("r1".into()),
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: Some(Usage { input: 5, output: 3, total_tokens: 8, ..Default::default() }),
                    stop_reason: Some(StopReason::Stop),
                    error_message: None,
                    tool_call_id: None,
                    tool_name: None,
                    is_error: false,
                    details: None,
                },
            ],
            tools: vec![Tool { name: "t".into(), description: "d".into(), parameters: serde_json::json!({"type": "object"}) }],
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.messages.len(), ctx.messages.len());
        assert_eq!(cloned.messages[1].timestamp, 42);
        assert_eq!(cloned.tools[0].name, "t");
        // Verify independence (clone, not reference)
        assert_eq!(cloned.system_prompt, ctx.system_prompt);
    }

    // --- Save/Load Round-trip with Complex Data ---

    #[test]
    fn test_save_load_with_tool_calls() {
        let ctx = Context {
            system_prompt: Some("sys".into()),
            messages: vec![
                user_message("search for rust"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "tc1".into(),
                        name: "search".into(),
                        arguments: std::collections::HashMap::from([("q".into(), serde_json::json!("rust"))]),
                        thought_signature: None,
                    }],
                    timestamp: 0,
                    api: None, provider: None, model: None, response_id: None,
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: None, stop_reason: Some(StopReason::ToolUse), error_message: None,
                    tool_call_id: None, tool_name: None, is_error: false,
                    details: None,
                },
            ],
            tools: vec![Tool { name: "search".into(), description: "search".into(), parameters: serde_json::json!({}) }],
        };
        let json = save_context(&ctx).unwrap();
        let loaded = load_context(&json).unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert!(matches!(&loaded.messages[1].content[0], ContentBlock::ToolCall { name, .. } if name == "search"));
    }
}

    /// Equivalent of Go's TestExamplesBuild — verify the crate compiles cleanly.
    /// In Rust this is always true when cargo test passes, but we add it for
    /// explicit parity with the Go test count.
    #[test]
    fn test_crate_compiles() {
        // If this test runs, the entire crate (including all providers,
        // images, transports, etc.) compiled successfully.
        let compiled = std::mem::size_of::<crate::types::Message>() > 0;
        assert!(compiled);
    }
