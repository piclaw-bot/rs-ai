//! Integration tests using the Faux provider to exercise the full streaming pipeline.

#[cfg(test)]
mod tests {
    use crate::provider::faux::*;
    use crate::harness::*;
    use crate::types::*;
    use crate::events::Event;
    use tokio_stream::StreamExt;
    use std::collections::HashMap;

    fn faux_model() -> Model {
        Model {
            id: "faux-model".into(),
            name: "Faux".into(),
            api: "faux".into(),
            provider: "faux".into(),
            base_url: "".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost { input: 3.0, output: 15.0, ..Default::default() },
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: None,
            compat: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_agent_loop_with_tool_call() {
        let model = faux_model();

        // Turn 1: model calls a tool
        let tool_response = "The file contains: hello world";

        // Simulate tool-call response
        let msg1 = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall {
                id: "tc1".into(),
                name: "read_file".into(),
                arguments: HashMap::from([("path".into(), serde_json::json!("test.txt"))]),
                thought_signature: None,
            }],
            timestamp: 0,
            api: Some("faux".into()),
            provider: Some("faux".into()),
            model: Some("faux-model".into()),
            response_id: Some("resp-1".into()),
            response_model: None,
            diagnostics: Vec::new(),
            usage: Some(Usage { input: 10, output: 5, total_tokens: 15, ..Default::default() }),
            stop_reason: Some(StopReason::ToolUse),
            error_message: None,
            tool_call_id: None,
            tool_name: None,
            is_error: false,
            details: None,
        };

        assert!(is_tool_use(&msg1));
        assert!(needs_tool_execution(&msg1));

        let calls = get_tool_calls(&msg1);
        assert_eq!(calls.len(), 1);
        if let ContentBlock::ToolCall { id, name, .. } = calls[0] {
            assert_eq!(id, "tc1");
            assert_eq!(name, "read_file");
        }

        // Build context with tool result
        let ctx = Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![user_message("Read test.txt")],
            tools: vec![Tool {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            }],
        };
        let ctx = append_assistant_message(ctx, &msg1);
        let ctx = append_tool_result(ctx, "tc1", "read_file", tool_response, false);
        assert_eq!(ctx.messages.len(), 3);

        // Turn 2: model responds with text
        let mut stream = stream_faux_text("Based on the file: hello world", &model);
        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }

        // Verify stream structure
        assert!(matches!(&events[0], Event::Start { .. }));
        assert!(matches!(&events[1], Event::TextStart));
        let text_deltas: String = events.iter().filter_map(|e| {
            if let Event::TextDelta { delta } = e { Some(delta.as_str()) } else { None }
        }).collect();
        assert!(text_deltas.contains("hello world"));
        assert!(matches!(events.last().unwrap(), Event::Done { .. }));

        if let Event::Done { message, reason } = events.last().unwrap() {
            assert_eq!(*reason, StopReason::Stop);
            let text = get_text_content(message);
            assert!(!text.is_empty());
            assert!(!is_tool_use(message));
        }
    }

    #[tokio::test]
    async fn test_streaming_harness() {
        let model = faux_model();
        let mut stream = stream_faux_text("Hello, world!", &model);

        let mut saw_start = false;
        let mut saw_text_start = false;
        let mut saw_text_end = false;
        let mut saw_done = false;
        let mut total_text = String::new();

        while let Some(evt) = stream.next().await {
            match evt {
                Event::Start { .. } => saw_start = true,
                Event::TextStart => saw_text_start = true,
                Event::TextDelta { delta } => total_text.push_str(&delta),
                Event::TextEnd => saw_text_end = true,
                Event::Done { .. } => saw_done = true,
                _ => {}
            }
        }

        assert!(saw_start);
        assert!(saw_text_start);
        assert!(saw_text_end);
        assert!(saw_done);
        assert_eq!(total_text, "Hello, world!");
    }

    #[tokio::test]
    async fn test_error_handling_harness() {
        let mut stream = stream_faux_error("simulated failure");
        let evt = stream.next().await.unwrap();
        match evt {
            Event::Error { reason, error, message } => {
                assert_eq!(reason, StopReason::Error);
                assert!(error.to_string().contains("simulated failure"));
                assert!(message.is_none());
            }
            _ => panic!("expected Error event"),
        }
    }

    #[tokio::test]
    async fn test_context_compaction_harness() {
        let mut ctx = Context {
            system_prompt: Some("System".into()),
            messages: (0..50).map(|i| user_message(&format!("msg {}", i))).collect(),
            tools: vec![],
        };
        assert!(ctx.messages.len() == 50);

        ctx = crate::compaction::compact_context(&ctx, 5, Some("earlier discussion about topics"));
        assert_eq!(ctx.messages.len(), 6); // summary + 5 recent
        assert!(ctx.system_prompt.is_some());

        // Verify summary is first
        if let ContentBlock::Text { text, .. } = &ctx.messages[0].content[0] {
            assert!(text.contains("summarized"));
        }
    }

    #[tokio::test]
    async fn test_hooks_harness() {
        let payload = serde_json::json!({"model": "test", "temperature": 0.7});

        // OnPayload hook modifies payload
        let modified = invoke_on_payload(payload.clone(), Some(&|mut p| {
            p["temperature"] = serde_json::json!(0.0);
            p["custom_field"] = serde_json::json!("added");
            p
        }));
        assert_eq!(modified["temperature"], 0.0);
        assert_eq!(modified["custom_field"], "added");

        // OnResponse hook receives status (verified by not panicking)
        let headers = HashMap::new();
        invoke_on_response(200, &headers, Some(&|_status, _| {
            // Hook invoked successfully
        }));
    }
}
