//! Final coverage tests: faux provider variants, context management, images.

#[cfg(test)]
mod tests {
    use crate::provider::faux::*;
    use crate::harness::*;
    use crate::types::*;
    use crate::events::Event;
    use crate::context::*;
    use crate::compaction::*;
    use tokio_stream::StreamExt;
    use std::collections::HashMap;

    fn faux_model() -> Model {
        Model {
            id: "faux".into(), name: "Faux".into(), api: "faux".into(),
            provider: "faux".into(), base_url: "".into(), reasoning: false,
            thinking_level_map: None, input: vec!["text".into()],
            cost: ModelCost { input: 1.0, output: 5.0, ..Default::default() },
            context_window: 128000, max_tokens: 4096, headers: None, api_key: None,
        }
    }

    // --- Faux Provider Tests ---

    #[tokio::test]
    async fn test_faux_text_stream() {
        let model = faux_model();
        let mut stream = stream_faux_text("Hello!", &model);
        let mut saw_text = false;
        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::TextDelta { .. } => saw_text = true,
                Event::Done { reason, message } => {
                    assert_eq!(reason, StopReason::Stop);
                    assert!(!get_text_content(&message).is_empty());
                    saw_done = true;
                }
                _ => {}
            }
        }
        assert!(saw_text && saw_done);
    }

    #[tokio::test]
    async fn test_faux_error() {
        let mut stream = stream_faux_error("test error");
        let evt = stream.next().await.unwrap();
        if let Event::Error { error, .. } = evt {
            assert!(error.to_string().contains("test error"));
        } else {
            panic!("expected error event");
        }
    }

    #[tokio::test]
    async fn test_faux_complete_flow() {
        let model = faux_model();
        let mut stream = stream_faux_text("Complete response", &model);
        let mut events: Vec<Event> = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }
        assert!(events.len() >= 4); // Start, TextStart, TextDelta(s), TextEnd, Done
        assert!(matches!(events.first().unwrap(), Event::Start { .. }));
        assert!(matches!(events.last().unwrap(), Event::Done { .. }));
    }

    // --- Context Management Tests ---

    #[test]
    fn test_clone_context() {
        let ctx = Context {
            system_prompt: Some("System".into()),
            messages: vec![user_message("Hello"), user_message("World")],
            tools: vec![Tool { name: "t".into(), description: "d".into(), parameters: serde_json::json!({}) }],
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.messages.len(), 2);
        assert_eq!(cloned.system_prompt, ctx.system_prompt);
        assert_eq!(cloned.tools.len(), 1);
    }

    #[test]
    fn test_context_json_round_trip() {
        let ctx = Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![
                user_message("hi"),
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text { text: "hello".into(), text_signature: None }],
                    timestamp: 123,
                    api: Some("openai-completions".into()),
                    provider: Some("openai".into()),
                    model: Some("gpt-4o".into()),
                    response_id: Some("resp-1".into()),
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
            tools: vec![],
        };
        let json = save_context(&ctx).unwrap();
        let loaded = load_context(&json).unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.system_prompt, ctx.system_prompt);
        assert_eq!(loaded.messages[1].response_id.as_deref(), Some("resp-1"));
    }

    #[test]
    fn test_compact_context_preserves_system_prompt() {
        let ctx = Context {
            system_prompt: Some("Important system prompt".into()),
            messages: (0..30).map(|i| user_message(&format!("msg {}", i))).collect(),
            tools: vec![],
        };
        let compacted = compact_context(&ctx, 5, Some("summary of earlier"));
        assert_eq!(compacted.system_prompt.as_deref(), Some("Important system prompt"));
        assert_eq!(compacted.messages.len(), 6); // summary + 5 recent
    }

    #[test]
    fn test_fits_in_context_window() {
        let model = faux_model();
        let ctx = Context {
            system_prompt: Some("short".into()),
            messages: vec![user_message("hi")],
            tools: vec![],
        };
        assert!(fits_in_context_window(&ctx, &model));
    }

    #[test]
    fn test_overflow_detection_normal() {
        let model = faux_model();
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "response".into(), text_signature: None }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: Some(Usage { input: 100, output: 50, total_tokens: 150, ..Default::default() }),
            stop_reason: Some(StopReason::Stop),
            error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        };
        assert!(!is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_overflow_detection_length_stop() {
        let model = faux_model();
        let msg = Message {
            role: Role::Assistant,
            content: vec![],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None,
            stop_reason: Some(StopReason::Length),
            error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        };
        assert!(is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_overflow_detection_error_message() {
        let model = faux_model();
        let msg = Message {
            role: Role::Assistant,
            content: vec![],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None,
            stop_reason: Some(StopReason::Error),
            error_message: Some("This model's maximum context length is 4096 tokens".into()),
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        };
        assert!(is_context_overflow(&msg, &model));
    }

    #[test]
    fn test_save_load_context_empty() {
        let ctx = Context { system_prompt: None, messages: vec![], tools: vec![] };
        let json = save_context(&ctx).unwrap();
        let loaded = load_context(&json).unwrap();
        assert!(loaded.messages.is_empty());
    }

    // --- Copilot Headers ---

    #[test]
    fn test_copilot_headers() {
        let h = crate::utils::copilot_headers();
        assert_eq!(h.get("Copilot-Integration-Id").unwrap(), "vscode-chat");
        assert!(h.contains_key("User-Agent"));
    }

    #[test]
    fn test_copilot_headers_with_intent() {
        let h = crate::utils::copilot_headers_with_intent("completion");
        assert_eq!(h.get("openai-intent").unwrap(), "completion");
        assert!(h.contains_key("User-Agent"));
    }

    // --- Token Estimation ---

    #[test]
    fn test_estimate_tokens_scales() {
        let short = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![],
        };
        let long = Context {
            system_prompt: Some("A very long system prompt that has many words in it.".into()),
            messages: vec![user_message("This is a much longer message with more content.")],
            tools: vec![],
        };
        assert!(estimate_tokens(&long) > estimate_tokens(&short));
    }

    // --- Models Are Equal ---

    #[test]
    fn test_models_are_equal_same() {
        let m = faux_model();
        assert!(models_are_equal(&m, &m));
    }

    #[test]
    fn test_models_are_equal_different() {
        let a = faux_model();
        let b = Model { id: "other".into(), ..a.clone() };
        assert!(!models_are_equal(&a, &b));
    }
}
