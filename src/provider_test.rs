//! Provider-level tests using mock HTTP servers.

#[cfg(test)]
mod tests {
    use crate::provider::openai::stream_openai;
    use crate::provider::anthropic::stream_anthropic;
    use crate::provider::mistral::stream_mistral;
    use crate::provider::responses::stream_responses;
    use crate::provider::codex::{build_codex_payload, replay_codex_ws_events};
    use crate::types::*;
    use crate::events::Event;
    use tokio_stream::StreamExt;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, path, header};

    fn test_model(api: &str, provider: &str, base_url: &str) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test".into(),
            api: api.into(),
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: Some("test-key".into()),
            compat: Default::default(),
        }
    }

    fn test_context() -> Context {
        Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![user_message("Hello")],
            tools: vec![],
        }
    }

    fn sse_response(events: &[&str]) -> String {
        events.iter().map(|e| format!("data: {}\n\n", e)).collect::<String>()
            + "data: [DONE]\n\n"
    }

    // --- OpenAI Tests ---

    #[tokio::test]
    async fn test_openai_stream_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"resp-1","choices":[{"delta":{"content":"Hello"},"index":0}]}"#,
                    r#"{"id":"resp-1","choices":[{"delta":{"content":" world"},"index":0}]}"#,
                    r#"{"id":"resp-1","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }

        // Verify event sequence
        assert!(matches!(&events[0], Event::Start { .. }));
        let text: String = events.iter().filter_map(|e| {
            if let Event::TextDelta { delta } = e { Some(delta.as_str()) } else { None }
        }).collect();
        assert_eq!(text, "Hello world");
        assert!(matches!(events.last().unwrap(), Event::Done { reason: StopReason::Stop, .. }));

        if let Event::Done { message, .. } = events.last().unwrap() {
            assert_eq!(message.response_id.as_deref(), Some("resp-1"));
            assert!(message.usage.is_some());
            assert_eq!(message.usage.as_ref().unwrap().input, 10);
        }
    }

    #[tokio::test]
    async fn test_openai_stream_tool_calls() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":""}}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut saw_tool_start = false;
        let mut tool_deltas = String::new();
        let mut stop_reason = None;

        while let Some(evt) = stream.next().await {
            match evt {
                Event::ToolCallStart { id, name } => {
                    assert_eq!(id, "tc1");
                    assert_eq!(name, "search");
                    saw_tool_start = true;
                }
                Event::ToolCallDelta { delta } => tool_deltas.push_str(&delta),
                Event::Done { reason, .. } => stop_reason = Some(reason),
                _ => {}
            }
        }

        assert!(saw_tool_start);
        assert!(tool_deltas.contains("\"q\":"));
        assert_eq!(stop_reason, Some(StopReason::ToolUse));
    }

    #[tokio::test]
    async fn test_openai_emits_tool_call_end() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{\"q\":\"rust\"}"}}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let mut tool_end = None;
        while let Some(evt) = stream.next().await {
            if let Event::ToolCallEnd { id, name, arguments } = evt {
                tool_end = Some((id, name, arguments));
            }
        }
        let (id, name, args) = tool_end.expect("ToolCallEnd must be emitted");
        assert_eq!(id, "tc1");
        assert_eq!(name, "search");
        assert_eq!(args["q"], "rust");
    }

    #[tokio::test]
    async fn test_azure_responses_uses_api_key_and_version() {
        use crate::provider::responses::stream_azure_responses;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("api-key", "test-key"))
            .and(wiremock::matchers::query_param("api-version", "v1"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"az-1\",\"model\":\"gpt-5\"}}\n\n\
                     data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"az-1\",\"model\":\"gpt-5\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("azure-openai-responses", "azure-openai-responses", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_azure_responses(&model, &ctx, &opts);
        let mut text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::TextDelta { delta } = evt { text.push_str(&delta); }
        }
        // Only matches if api-key header + api-version query were present.
        assert_eq!(text, "hi");
    }

    #[tokio::test]
    async fn test_responses_stream_tool_calls() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-2\",\"model\":\"gpt-4.1\"}}\n\n\
                     data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"\"}}\n\n\
                     data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"q\\\":\"}\n\n\
                     data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"q\\\":\\\"rust\\\"}\"}\n\n\
                     data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"rust\\\"}\"}}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-2\",\"model\":\"gpt-4.1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4,\"total_tokens\":14}}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);

        let mut saw_tool_start = false;
        let mut saw_tool_end = false;
        let mut deltas = String::new();
        let mut done_reason = None;

        while let Some(evt) = stream.next().await {
            match evt {
                Event::ToolCallStart { id, name } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "search");
                    saw_tool_start = true;
                }
                Event::ToolCallDelta { delta } => deltas.push_str(&delta),
                Event::ToolCallEnd { id, name, arguments } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "search");
                    assert_eq!(arguments["q"], "rust");
                    saw_tool_end = true;
                }
                Event::Done { reason, message } => {
                    done_reason = Some(reason);
                    assert_eq!(message.response_id.as_deref(), Some("resp-2"));
                    assert_eq!(message.response_model.as_deref(), Some("gpt-4.1"));
                    assert!(message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { name, .. } if name == "search")));
                }
                _ => {}
            }
        }

        assert!(saw_tool_start);
        assert!(saw_tool_end);
        assert!(deltas.contains("\"q\":"));
        assert_eq!(done_reason, Some(StopReason::ToolUse));
    }

    #[tokio::test]
    async fn test_openai_stream_reasoning_and_final_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"resp-r1","choices":[{"delta":{"reasoning_content":"think-1"},"index":0}]}"#,
                    r#"{"id":"resp-r1","choices":[{"delta":{"content":"answer"},"index":0}]}"#,
                    r#"{"id":"resp-r1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{\"q\":\"rust\"}"}}]},"index":0}]}"#,
                    r#"{"id":"resp-r1","choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut saw_thinking = false;
        let mut done: Option<Message> = None;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::ThinkingDelta { delta } => {
                    assert_eq!(delta, "think-1");
                    saw_thinking = true;
                }
                Event::Done { message, .. } => done = Some(message),
                _ => {}
            }
        }

        let message = done.expect("done message");
        assert!(saw_thinking);
        assert!(message.content.iter().any(|b| matches!(b, ContentBlock::Thinking { thinking, .. } if thinking == "think-1")));
        assert!(message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { id, name, .. } if id == "tc1" && name == "search")));
    }

    #[tokio::test]
    async fn test_responses_stream_reasoning_blocks() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-r3\",\"model\":\"gpt-5\"}}\n\n\
                     data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\"}}\n\n\
                     data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"step one\"}\n\n\
                     data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"content\":[{\"text\":\"step one\"}]}}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-r3\",\"model\":\"gpt-5\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":6}}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);

        let mut saw_thinking = false;
        let mut done: Option<Message> = None;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::ThinkingDelta { delta } => {
                    assert_eq!(delta, "step one");
                    saw_thinking = true;
                }
                Event::Done { message, .. } => done = Some(message),
                _ => {}
            }
        }

        let message = done.expect("done message");
        assert!(saw_thinking);
        assert!(message.content.iter().any(|b| matches!(b, ContentBlock::Thinking { thinking, .. } if thinking == "step one")));
        assert_eq!(message.response_model.as_deref(), Some("gpt-5"));
    }

    #[test]
    fn test_openai_thinking_signature_replay() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "openai", "https://example.com") };
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking { thinking: "reasoned".into(), thinking_signature: Some("reasoning_content".into()), redacted: false },
                    ContentBlock::Text { text: "answer".into(), text_signature: None },
                ],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        assert_eq!(payload["messages"][0]["reasoning_content"], "reasoned");
        assert_eq!(payload["messages"][0]["content"], "answer");
    }

    #[test]
    fn test_openai_tool_call_reasoning_details_replay() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "openrouter", "https://openrouter.ai/api/v1") };
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall {
                    id: "tc1".into(),
                    name: "search".into(),
                    arguments: std::collections::HashMap::from([("q".to_string(), serde_json::json!("rust"))]),
                    thought_signature: Some("{\"type\":\"reasoning.text\",\"text\":\"why\"}".into()),
                }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::ToolUse), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        let details = &payload["messages"][0]["reasoning_details"];
        assert_eq!(details[0]["type"], "reasoning.text");
        assert_eq!(details[0]["text"], "why");
    }

    #[test]
    fn test_openai_thinking_as_text() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "openai", "https://example.com") };
        let overrides = crate::compat::OpenAICompletionsCompat { requires_thinking_as_text: Some(true), ..Default::default() };
        let compat = crate::compat::detect_compat_for_model(&model, Some(&overrides));
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking { thinking: "reasoned".into(), thinking_signature: None, redacted: false },
                    ContentBlock::Text { text: "answer".into(), text_signature: None },
                ],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &compat);
        let content = &payload["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["text"], "reasoned");
        assert_eq!(content[1]["text"], "answer");
    }

    #[test]
    fn test_openai_normalizes_responses_tool_call_id() {
        let model = test_model("openai-completions", "openai", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall {
                    id: "call_abc|fc_verylongidentifierthatissuperlongandexceeds40chars+/=".into(),
                    name: "search".into(),
                    arguments: std::collections::HashMap::new(),
                    thought_signature: None,
                }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::ToolUse), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        let id = payload["messages"][0]["tool_calls"][0]["id"].as_str().unwrap();
        assert_eq!(id, "call_abc");
        assert!(!id.contains('|'));
    }

    #[test]
    fn test_openai_deepseek_reasoning_content_on_assistant() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "deepseek", "https://api.deepseek.com") };
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text { text: "hi".into(), text_signature: None }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        assert_eq!(payload["messages"][0]["reasoning_content"], "");
    }

    #[test]
    fn test_openai_qwen_chat_template_thinking() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "openai", "https://example.com") };
        let overrides = crate::compat::OpenAICompletionsCompat { thinking_format: Some("qwen-chat-template".into()), ..Default::default() };
        let compat = crate::compat::detect_compat_for_model(&model, Some(&overrides));
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &compat);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(payload["chat_template_kwargs"]["preserve_thinking"], true);
    }

    #[test]
    fn test_openai_string_thinking_format() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "openai", "https://example.com") };
        let overrides = crate::compat::OpenAICompletionsCompat { thinking_format: Some("string-thinking".into()), ..Default::default() };
        let compat = crate::compat::detect_compat_for_model(&model, Some(&overrides));
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::Medium), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &compat);
        assert_eq!(payload["thinking"], "medium");
    }

    #[test]
    fn test_openai_zai_thinking_format() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "zai", "https://z.ai/api") };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &crate::compat::detect_compat(&model));
        // zai must use a thinking object, not enable_thinking.
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert!(payload.get("enable_thinking").is_none());
    }

    #[test]
    fn test_openai_deepseek_thinking_object() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "deepseek", "https://api.deepseek.com") };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &crate::compat::detect_compat(&model));
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert_eq!(payload["reasoning_effort"], "high");
    }

    #[test]
    fn test_openai_tool_choice_passthrough() {
        let model = test_model("openai-completions", "openai", "https://example.com");
        let ctx = test_context();
        let opts = StreamOptions { tool_choice: Some(serde_json::json!("required")), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &crate::compat::detect_compat(&model));
        assert_eq!(payload["tool_choice"], "required");
    }

    #[test]
    fn test_openai_reasoning_dropped_for_non_reasoning_model() {
        // test_model has reasoning: false → supported levels = [off] → reasoning clamped away.
        let model = test_model("openai-completions", "deepseek", "https://api.deepseek.com");
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &crate::compat::detect_compat(&model));
        assert!(payload.get("reasoning_effort").is_none());
        assert!(payload.get("reasoning").is_none());
    }

    #[test]
    fn test_openai_reasoning_kept_for_reasoning_model() {
        let model = Model { reasoning: true, ..test_model("openai-completions", "deepseek", "https://api.deepseek.com") };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &opts, &crate::compat::detect_compat(&model));
        assert_eq!(payload["reasoning_effort"], "high");
    }

    #[test]
    fn test_openai_build_payload_downgrades_images_for_text_model() {
        let model = test_model("openai-completions", "openai", "https://example.com");
        // test_model has input = ["text"] only → non-vision.
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text { text: "see".into(), text_signature: None },
                    ContentBlock::Image { data: "abc".into(), mime_type: "image/png".into() },
                ],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: None, error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        let serialized = serde_json::to_string(&payload).unwrap();
        assert!(!serialized.contains("image_url"), "image must be downgraded for non-vision model");
        assert!(serialized.contains("omitted"));
    }

    #[test]
    fn test_openai_build_payload_preserves_assistant_tool_calls() {
        let model = test_model("openai-completions", "openai", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall {
                    id: "tc1".into(),
                    name: "search".into(),
                    arguments: std::collections::HashMap::from([("q".into(), serde_json::json!("rust"))]),
                    thought_signature: None,
                }],
                timestamp: 0,
                api: None,
                provider: None,
                model: None,
                response_id: None,
                response_model: None,
                diagnostics: Vec::new(),
                usage: None,
                stop_reason: Some(StopReason::ToolUse),
                error_message: None,
                tool_call_id: None,
                tool_name: None,
                is_error: false,
                details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::openai::build_payload(&model, &ctx, &StreamOptions::default(), &crate::compat::detect_compat(&model));
        let msg = &payload["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert!(msg["tool_calls"].is_array());
        assert_eq!(msg["tool_calls"][0]["id"], "tc1");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "search");
    }

    #[test]
    fn test_responses_assistant_ordering_and_output_text() {
        let model = test_model("openai-responses", "openai", "https://api.openai.com/v1");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking { thinking: "r".into(), thinking_signature: Some("{\"type\":\"reasoning\",\"id\":\"rs_1\"}".into()), redacted: false },
                    ContentBlock::Text { text: "answer".into(), text_signature: None },
                    ContentBlock::ToolCall { id: "call_1|fc_1".into(), name: "t".into(), arguments: std::collections::HashMap::new(), thought_signature: None },
                ],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::ToolUse), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        let input = payload["input"].as_array().unwrap();
        // Order: reasoning item, then message (output_text), then function_call.
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "answer");
        assert_eq!(input[2]["type"], "function_call");
    }

    #[test]
    fn test_responses_assistant_text_id_and_phase() {
        let model = test_model("openai-responses", "openai", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    // Signed text block carries an explicit id + phase.
                    ContentBlock::Text { text: "hi".into(), text_signature: Some("{\"v\":1,\"id\":\"msg_abc\",\"phase\":\"final_answer\"}".into()) },
                    // Unsigned second text block falls back to a deterministic msg_pi id.
                    ContentBlock::Text { text: "more".into(), text_signature: None },
                ],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        let input = payload["input"].as_array().unwrap();
        assert_eq!(input[0]["id"], "msg_abc");
        assert_eq!(input[0]["phase"], "final_answer");
        assert_eq!(input[1]["id"], "msg_pi_0_1");
        assert!(input[1].get("phase").is_none());
    }

    #[test]
    fn test_responses_foreign_tool_call_id_normalized() {
        // A tool call captured from a different provider/api gets a hashed fc_ item id.
        let model = test_model("openai-responses", "openai", "https://example.com");
        let mut msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall {
                id: "call_1|weird id!".into(),
                name: "t".into(),
                arguments: std::collections::HashMap::new(),
                thought_signature: None,
            }],
            timestamp: 0,
            api: Some("anthropic-messages".into()), provider: Some("anthropic".into()), model: Some("claude".into()),
            response_id: None, response_model: None, diagnostics: Vec::new(), usage: None,
            stop_reason: Some(StopReason::ToolUse), error_message: None,
            tool_call_id: None, tool_name: None, is_error: false, details: None,
        };
        let ctx = Context { system_prompt: None, messages: vec![msg.clone()], tools: vec![] };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        let fc = payload["input"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["call_id"], "call_1");
        assert!(fc["id"].as_str().unwrap().starts_with("fc_"));

        // Same provider/api but different model -> item id omitted to avoid pairing validation.
        msg.provider = Some("openai".into());
        msg.api = Some("openai-responses".into());
        msg.model = Some("gpt-other".into());
        if let ContentBlock::ToolCall { id, .. } = &mut msg.content[0] { *id = "call_1|fc_abc".into(); }
        let ctx = Context { system_prompt: None, messages: vec![msg], tools: vec![] };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        let fc = payload["input"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["call_id"], "call_1");
        assert!(fc["id"].is_null());
    }

    #[test]
    fn test_responses_build_payload_preserves_tool_history() {
        let model = test_model("openai-responses", "openai", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "call_1|fc_1".into(),
                        name: "search".into(),
                        arguments: std::collections::HashMap::from([("q".into(), serde_json::json!("rust"))]),
                        thought_signature: None,
                    }],
                    timestamp: 0,
                    api: None,
                    provider: None,
                    model: None,
                    response_id: None,
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: None,
                    stop_reason: Some(StopReason::ToolUse),
                    error_message: None,
                    tool_call_id: None,
                    tool_name: None,
                    is_error: false,
                    details: None,
                },
                Message {
                    role: Role::ToolResult,
                    content: vec![ContentBlock::Text { text: "done".into(), text_signature: None }],
                    timestamp: 0,
                    api: None,
                    provider: None,
                    model: None,
                    response_id: None,
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: None,
                    stop_reason: None,
                    error_message: None,
                    tool_call_id: Some("call_1|fc_1".into()),
                    tool_name: Some("search".into()),
                    is_error: false,
                    details: None,
                }
            ],
            tools: vec![],
        };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        assert!(payload["input"].is_array());
        let arr = payload["input"].as_array().unwrap();
        assert!(arr.iter().any(|v| v["type"] == "function_call" && v["call_id"] == "call_1" && v["name"] == "search"));
        assert!(arr.iter().any(|v| v["type"] == "function_call_output" && v["call_id"] == "call_1" && v["output"] == "done"));
    }

    #[test]
    fn test_codex_build_payload_matches_responses_history_shape() {
        let model = test_model("openai-codex-responses", "openai", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "call_1|fc_1".into(),
                        name: "search".into(),
                        arguments: std::collections::HashMap::from([("q".into(), serde_json::json!("rust"))]),
                        thought_signature: None,
                    }],
                    timestamp: 0,
                    api: None,
                    provider: None,
                    model: None,
                    response_id: None,
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: None,
                    stop_reason: Some(StopReason::ToolUse),
                    error_message: None,
                    tool_call_id: None,
                    tool_name: None,
                    is_error: false,
                    details: None,
                },
                Message {
                    role: Role::ToolResult,
                    content: vec![ContentBlock::Text { text: "done".into(), text_signature: None }],
                    timestamp: 0,
                    api: None,
                    provider: None,
                    model: None,
                    response_id: None,
                    response_model: None,
                    diagnostics: Vec::new(),
                    usage: None,
                    stop_reason: None,
                    error_message: None,
                    tool_call_id: Some("call_1|fc_1".into()),
                    tool_name: Some("search".into()),
                    is_error: false,
                    details: None,
                }
            ],
            tools: vec![],
        };
        let payload = build_codex_payload(&model, &ctx, &StreamOptions::default());
        let arr = payload["input"].as_array().unwrap();
        assert!(arr.iter().any(|v| v["type"] == "function_call" && v["call_id"] == "call_1"));
        assert!(arr.iter().any(|v| v["type"] == "function_call_output" && v["call_id"] == "call_1"));
    }

    #[test]
    fn test_codex_payload_structure() {
        use crate::provider::codex::build_codex_payload;
        let model = Model { reasoning: true, ..test_model("openai-codex-responses", "openai", "https://chatgpt.com/backend-api") };
        let ctx = Context {
            system_prompt: Some("custom sys".into()),
            messages: vec![user_message("hi")],
            tools: vec![],
        };
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::Medium), session_id: Some("s1".into()), ..Default::default() };
        let payload = build_codex_payload(&model, &ctx, &opts);
        assert_eq!(payload["store"], false);
        assert_eq!(payload["instructions"], "custom sys");
        assert_eq!(payload["tool_choice"], "auto");
        assert_eq!(payload["parallel_tool_calls"], true);
        assert_eq!(payload["text"]["verbosity"], "low");
        assert_eq!(payload["prompt_cache_key"], "s1");
        assert_eq!(payload["reasoning"]["summary"], "auto");
        // System prompt must NOT be duplicated in input.
        let input = payload["input"].as_array().unwrap();
        assert!(!input.iter().any(|m| matches!(m.get("role").and_then(|r| r.as_str()), Some("system") | Some("developer"))));
    }

    #[test]
    fn test_codex_payload_tool_strict_null() {
        use crate::provider::codex::build_codex_payload;
        let model = test_model("openai-codex-responses", "openai", "https://chatgpt.com/backend-api");
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "t".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) }],
        };
        let payload = build_codex_payload(&model, &ctx, &StreamOptions::default());
        assert!(payload["tools"][0]["strict"].is_null());
    }

    #[test]
    fn test_codex_ws_error_event() {
        let model = test_model("openai-codex-responses", "openai", "https://example.com");
        let events = vec![
            serde_json::json!({"type":"response.created","response":{"id":"c1","model":"codex"}}),
            serde_json::json!({"type":"error","code":"rate_limited","message":"slow down"}),
        ];
        let replayed = replay_codex_ws_events(&model, &events);
        let err = replayed.iter().find_map(|e| match e { Event::Error { error, .. } => Some(error.to_string()), _ => None });
        assert!(err.unwrap().contains("slow down"));
        // No Done event after an error.
        assert!(!replayed.iter().any(|e| matches!(e, Event::Done { .. })));
    }

    #[test]
    fn test_codex_ws_event_replay_tool_and_reasoning() {
        let model = test_model("openai-codex-responses", "openai", "https://example.com");
        let events = vec![
            serde_json::json!({"type":"response.created","response":{"id":"resp-c1","model":"codex-mini"}}),
            serde_json::json!({"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}),
            serde_json::json!({"type":"response.reasoning_text.delta","delta":"ponder"}),
            serde_json::json!({"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","content":[{"text":"ponder"}]}}),
            serde_json::json!({"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"search","arguments":""}}),
            serde_json::json!({"type":"response.function_call_arguments.delta","delta":"{\"q\":"}),
            serde_json::json!({"type":"response.function_call_arguments.done","arguments":"{\"q\":\"rust\"}"}),
            serde_json::json!({"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"search","arguments":"{\"q\":\"rust\"}"}}),
            serde_json::json!({"type":"response.output_text.delta","delta":"answer"}),
            serde_json::json!({"type":"response.completed","response":{"id":"resp-c1","model":"codex-mini","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}),
        ];
        let replayed = replay_codex_ws_events(&model, &events);
        assert!(replayed.iter().any(|e| matches!(e, Event::ThinkingDelta { delta } if delta == "ponder")));
        assert!(replayed.iter().any(|e| matches!(e, Event::ToolCallEnd { id, name, arguments } if id == "call_1" && name == "search" && arguments["q"] == "rust")));
        let done = replayed.iter().find_map(|e| match e { Event::Done { message, .. } => Some(message), _ => None }).expect("done");
        assert_eq!(done.response_id.as_deref(), Some("resp-c1"));
        assert_eq!(done.response_model.as_deref(), Some("codex-mini"));
        assert!(done.content.iter().any(|b| matches!(b, ContentBlock::Thinking { thinking, .. } if thinking == "ponder")));
        assert!(done.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { name, .. } if name == "search")));
        assert!(done.content.iter().any(|b| matches!(b, ContentBlock::Text { text, .. } if text == "answer")));
    }

    #[test]
    fn test_responses_reasoning_effort_maps_thinking_level() {
        use std::collections::HashMap;
        let model = Model {
            reasoning: true,
            thinking_level_map: Some(HashMap::from([("high".to_string(), Some("xhigh".to_string()))])),
            ..test_model("openai-responses", "openai", "https://api.openai.com/v1")
        };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &opts);
        // The thinkingLevelMap remaps high -> xhigh.
        assert_eq!(payload["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn test_responses_reasoning_model_uses_developer_role() {
        let model = Model { reasoning: true, ..test_model("openai-responses", "openai", "https://api.openai.com/v1") };
        let ctx = Context { system_prompt: Some("sys".into()), messages: vec![user_message("hi")], tools: vec![] };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &StreamOptions::default());
        assert_eq!(payload["input"][0]["role"], "developer");
        // Non-reasoning model uses system.
        let model2 = Model { reasoning: false, ..test_model("openai-responses", "openai", "https://api.openai.com/v1") };
        let payload2 = crate::provider::responses::build_responses_payload(&model2, &ctx, &StreamOptions::default());
        assert_eq!(payload2["input"][0]["role"], "system");
    }

    #[test]
    fn test_responses_build_payload_includes_cache_and_store_flags() {
        let model = test_model("openai-responses", "openai", "https://api.openai.com/v1");
        let ctx = test_context();
        let opts = StreamOptions {
            session_id: Some("sess-1".into()),
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        };
        let payload = crate::provider::responses::build_responses_payload(&model, &ctx, &opts);
        assert_eq!(payload["store"], false);
        // session_id is sent via headers, not the body.
        assert!(payload.get("session_id").is_none());
        assert_eq!(payload["prompt_cache_key"], "sess-1");
        assert_eq!(payload["prompt_cache_retention"], "24h");
    }

    #[tokio::test]
    async fn test_anthropic_error_event_emits_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
                     event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"server overloaded\"}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("anthropic-messages", "anthropic", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_anthropic(&model, &ctx, &opts);
        let mut err = None;
        while let Some(evt) = stream.next().await {
            if let Event::Error { error, .. } = evt { err = Some(error.to_string()); }
        }
        assert!(err.unwrap().contains("overloaded"));
    }

    #[tokio::test]
    async fn test_responses_incomplete_status_is_length() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\"}}\n\n\
                     data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);
        let mut reason = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { reason: r, .. } = evt { reason = Some(r); }
        }
        assert_eq!(reason, Some(StopReason::Length));
    }

    #[tokio::test]
    async fn test_responses_failed_status_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\"}}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\",\"status\":\"failed\",\"error\":{\"message\":\"internal\"}}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);
        let mut saw_error = false;
        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            match evt { Event::Error { .. } => saw_error = true, Event::Done { .. } => saw_done = true, _ => {} }
        }
        assert!(saw_error && !saw_done);
    }

    #[tokio::test]
    async fn test_responses_error_event_emits_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5\"}}\n\n\
                     data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"boom\"}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);
        let mut err = None;
        while let Some(evt) = stream.next().await {
            if let Event::Error { error, .. } = evt { err = Some(error.to_string()); }
        }
        let m = err.unwrap();
        assert!(m.contains("boom") && m.contains("server_error"));
    }

    #[tokio::test]
    async fn test_openai_in_band_error_chunk_emits_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"error":{"message":"rate limited by provider","code":429}}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let mut err_msg = None;
        while let Some(evt) = stream.next().await {
            if let Event::Error { error, .. } = evt { err_msg = Some(error.to_string()); }
        }
        assert!(err_msg.unwrap().contains("rate limited"));
    }

    #[tokio::test]
    async fn test_openai_stream_without_finish_reason_is_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                // content but no finish_reason, then [DONE]
                .set_body_string(sse_response(&[
                    r#"{"id":"r1","choices":[{"delta":{"content":"partial"},"index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let mut saw_error = false;
        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::Error { .. } => saw_error = true,
                Event::Done { .. } => saw_done = true,
                _ => {}
            }
        }
        assert!(saw_error, "stream without finish_reason must emit an error");
        assert!(!saw_done);
    }

    #[tokio::test]
    async fn test_openai_retries_on_503_then_succeeds() {
        let server = MockServer::start().await;
        // First attempt: 503. Second: 200.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("busy"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"r1","choices":[{"delta":{"content":"ok"},"index":0}]}"#,
                    r#"{"id":"r1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions { max_retries: Some(2), max_retry_delay_ms: Some(5), ..Default::default() };
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let mut text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::TextDelta { delta } = evt { text.push_str(&delta); }
        }
        assert_eq!(text, "ok");
    }

    #[tokio::test]
    async fn test_openai_missing_api_key() {
        let model = Model {
            api_key: None,
            ..test_model("openai-completions", "openai", "http://localhost:1")
        };
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let evt = stream.next().await.unwrap();
        assert!(matches!(evt, Event::Error { .. }));
    }

    #[tokio::test]
    async fn test_openai_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut saw_error = false;
        while let Some(evt) = stream.next().await {
            if matches!(evt, Event::Error { .. }) {
                saw_error = true;
            }
        }
        assert!(saw_error);
    }

    #[tokio::test]
    async fn test_github_copilot_dynamic_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("X-Initiator", "user"))
            .and(header("Openai-Intent", "conversation-edits"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"c1","choices":[{"delta":{"content":"hi"},"index":0}]}"#,
                    r#"{"id":"c1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "github-copilot", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);
        let mut text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::TextDelta { delta } = evt { text.push_str(&delta); }
        }
        // If headers were missing, the mock wouldn't match and we'd get no text.
        assert_eq!(text, "hi");
    }

    // --- Anthropic Tests ---

    #[tokio::test]
    async fn test_anthropic_stream_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-1\",\"usage\":{\"input_tokens\":15,\"output_tokens\":0}}}\n\n\
                     event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\"}}\n\n\
                     event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi there!\"}}\n\n\
                     event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n\
                     event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("anthropic-messages", "anthropic", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_anthropic(&model, &ctx, &opts);

        let mut text = String::new();
        let mut saw_done = false;

        while let Some(evt) = stream.next().await {
            match evt {
                Event::TextDelta { delta } => text.push_str(&delta),
                Event::Done { reason, message } => {
                    assert_eq!(reason, StopReason::Stop);
                    assert_eq!(message.response_id.as_deref(), Some("msg-1"));
                    saw_done = true;
                }
                _ => {}
            }
        }

        assert_eq!(text, "Hi there!");
        assert!(saw_done);
    }

    #[tokio::test]
    async fn test_anthropic_message_delta_updates_cache_usage() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":20,\"output_tokens\":0,\"cache_read_input_tokens\":4}}}\n\n\
                     event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"text\"}}\n\n\
                     event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}\n\n\
                     event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n\
                     event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7,\"cache_creation_input_tokens\":9}}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("anthropic-messages", "anthropic", &server.uri());
        let ctx = test_context();
        let opts = StreamOptions::default();
        let mut stream = stream_anthropic(&model, &ctx, &opts);
        let mut usage = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { message, .. } = evt { usage = message.usage; }
        }
        let u = usage.expect("usage");
        // input preserved from message_start; output + cache_creation from message_delta;
        // cache_read preserved from message_start.
        assert_eq!(u.input, 20);
        assert_eq!(u.output, 7);
        assert_eq!(u.cache_read, 4);
        assert_eq!(u.cache_write, 9);
        assert_eq!(u.total_tokens, 40);
    }

    #[tokio::test]
    async fn test_anthropic_tool_use_stream() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-2\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n\
                     event: content_block_start\ndata: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"tc1\",\"name\":\"search\"}}\n\n\
                     event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n\
                     event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"rust\\\"}\"}}\n\n\
                     event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n\
                     event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("anthropic-messages", "anthropic", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_anthropic(&model, &ctx, &opts);

        let mut saw_tool_start = false;
        let mut tool_json = String::new();
        let mut stop_reason = None;
        let mut done_msg: Option<Message> = None;

        while let Some(evt) = stream.next().await {
            match evt {
                Event::ToolCallStart { id, name } => {
                    assert_eq!(id, "tc1");
                    assert_eq!(name, "search");
                    saw_tool_start = true;
                }
                Event::ToolCallDelta { delta } => tool_json.push_str(&delta),
                Event::Done { reason, message } => { stop_reason = Some(reason); done_msg = Some(message); }
                _ => {}
            }
        }

        assert!(saw_tool_start);
        assert_eq!(tool_json, "{\"q\":\"rust\"}");
        assert_eq!(stop_reason, Some(StopReason::ToolUse));
        let msg = done_msg.expect("done message");
        assert!(msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { id, name, arguments, .. } if id == "tc1" && name == "search" && arguments["q"] == "rust")));
    }

    #[test]
    fn test_anthropic_redacted_thinking_payload_roundtrip() {
        use crate::provider::anthropic::build_anthropic_payload;
        let model = test_model("anthropic-messages", "anthropic", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Thinking { thinking: "[Reasoning redacted]".into(), thinking_signature: Some("opaque-blob".into()), redacted: true }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = build_anthropic_payload(&model, &ctx, &StreamOptions::default());
        let block = &payload["messages"][0]["content"][0];
        assert_eq!(block["type"], "redacted_thinking");
        assert_eq!(block["data"], "opaque-blob");
    }

    #[test]
    fn test_anthropic_adaptive_thinking_uses_effort_not_budget() {
        use crate::provider::anthropic::build_anthropic_payload;
        // Adaptive-thinking models (forceAdaptiveThinking) send an effort, not a token budget.
        let mut model = test_model("anthropic-messages", "anthropic", "https://api.anthropic.com");
        model.reasoning = true;
        model.compat.force_adaptive_thinking = Some(true);
        // xhigh maps to "max" on Opus 4.6 via thinkingLevelMap.
        model.thinking_level_map = Some(std::collections::HashMap::from([
            ("xhigh".to_string(), Some("max".to_string())),
        ]));
        let ctx = Context { system_prompt: None, messages: vec![user_message("hi")], tools: vec![] };

        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = build_anthropic_payload(&model, &ctx, &opts);
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert_eq!(payload["thinking"]["effort"], "high");
        assert!(payload["thinking"].get("budget_tokens").is_none());

        let opts = StreamOptions { reasoning: Some(ThinkingLevel::XHigh), ..Default::default() };
        let payload = build_anthropic_payload(&model, &ctx, &opts);
        assert_eq!(payload["thinking"]["effort"], "max");
    }

    #[test]
    fn test_anthropic_budget_thinking_for_non_adaptive() {
        use crate::provider::anthropic::build_anthropic_payload;
        let mut model = test_model("anthropic-messages", "anthropic", "https://api.anthropic.com");
        model.reasoning = true;
        let ctx = Context { system_prompt: None, messages: vec![user_message("hi")], tools: vec![] };
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let payload = build_anthropic_payload(&model, &ctx, &opts);
        assert_eq!(payload["thinking"]["type"], "enabled");
        assert!(payload["thinking"].get("budget_tokens").is_some());
        assert!(payload["thinking"].get("effort").is_none());
    }

    #[test]
    fn test_anthropic_oauth_identity_system_block() {
        use crate::provider::anthropic::build_anthropic_payload;
        let mut model = test_model("anthropic-messages", "anthropic", "https://api.anthropic.com");
        model.api_key = Some("sk-ant-oat01-abc".into());
        let ctx = Context {
            system_prompt: Some("be helpful".into()),
            messages: vec![user_message("hi")],
            tools: vec![],
        };
        let payload = build_anthropic_payload(&model, &ctx, &StreamOptions::default());
        let system = payload["system"].as_array().unwrap();
        assert!(system[0]["text"].as_str().unwrap().contains("Claude Code"));
        assert_eq!(system[1]["text"], "be helpful");
    }

    #[test]
    fn test_anthropic_merges_consecutive_tool_results() {
        use crate::provider::anthropic::build_anthropic_payload;
        let model = test_model("anthropic-messages", "anthropic", "https://example.com");
        let tr = |id: &str, txt: &str| Message {
            role: Role::ToolResult,
            content: vec![ContentBlock::Text { text: txt.into(), text_signature: None }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None, diagnostics: Vec::new(), usage: None,
            stop_reason: None, error_message: None,
            tool_call_id: Some(id.into()), tool_name: Some("t".into()),
            is_error: false, details: None,
        };
        let ctx = Context {
            system_prompt: Some("sys".into()),
            messages: vec![tr("a", "1"), tr("b", "2")],
            tools: vec![],
        };
        let opts = StreamOptions { cache_retention: Some(CacheRetention::Long), ..Default::default() };
        let payload = build_anthropic_payload(&model, &ctx, &opts);
        // Two consecutive tool results must collapse into ONE user message with two blocks.
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
        let blocks = payload["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["tool_use_id"], "a");
        assert_eq!(blocks[1]["tool_use_id"], "b");
        // System prompt is structured with cache_control.
        assert_eq!(payload["system"][0]["type"], "text");
        assert_eq!(payload["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(payload["system"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn test_anthropic_tool_result_payload_shape() {
        use crate::provider::anthropic::build_anthropic_payload;
        let model = test_model("anthropic-messages", "anthropic", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::ToolResult,
                content: vec![ContentBlock::Text { text: "42".into(), text_signature: None }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: None, error_message: None,
                tool_call_id: Some("tc1".into()), tool_name: Some("calc".into()),
                is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = build_anthropic_payload(&model, &ctx, &StreamOptions::default());
        let block = &payload["messages"][0]["content"][0];
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "tc1");
        assert_eq!(block["content"][0]["text"], "42");
    }

    #[tokio::test]
    async fn test_anthropic_refusal_stop_reason() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-9\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
                     event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"explanation\":\"nope\"}}}\n\n\
                     event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("anthropic-messages", "anthropic", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_anthropic(&model, &ctx, &opts);
        let mut done: Option<Message> = None;
        let mut reason = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { reason: r, message } = evt { reason = Some(r); done = Some(message); }
        }
        assert_eq!(reason, Some(StopReason::Error));
        assert_eq!(done.unwrap().error_message.as_deref(), Some("nope"));
    }

    // --- Mistral Tests ---

    #[tokio::test]
    async fn test_mistral_stream_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"cmpl-1","choices":[{"delta":{"content":"Bonjour"},"index":0}]}"#,
                    r#"{"id":"cmpl-1","choices":[{"delta":{},"finish_reason":"stop","index":0}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("mistral-conversations", "mistral", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_mistral(&model, &ctx, &opts);

        let mut text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::TextDelta { delta } = evt {
                text.push_str(&delta);
            }
        }
        assert_eq!(text, "Bonjour");
    }

    #[tokio::test]
    async fn test_mistral_streams_thinking_array_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"pondering"}]}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{"content":[{"type":"text","text":"answer"}]},"index":0}]}"#,
                    r#"{"choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("mistral-conversations", "mistral", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_mistral(&model, &ctx, &opts);
        let mut thinking = String::new();
        let mut text = String::new();
        let mut done: Option<Message> = None;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::ThinkingDelta { delta } => thinking.push_str(&delta),
                Event::TextDelta { delta } => text.push_str(&delta),
                Event::Done { message, .. } => done = Some(message),
                _ => {}
            }
        }
        assert_eq!(thinking, "pondering");
        assert_eq!(text, "answer");
        let msg = done.unwrap();
        assert!(msg.content.iter().any(|b| matches!(b, ContentBlock::Thinking { thinking, .. } if thinking == "pondering")));
    }

    #[tokio::test]
    async fn test_mistral_stream_tool_calls() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(sse_response(&[
                    r#"{"id":"cmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"search","arguments":"{\"q\":"}}]},"index":0}]}"#,
                    r#"{"id":"cmpl-2","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"rust\"}"}}]},"index":0}]}"#,
                    r#"{"id":"cmpl-2","choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
                ]))
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("mistral-conversations", "mistral", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_mistral(&model, &ctx, &opts);

        let mut done: Option<Message> = None;
        let mut reason = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { reason: r, message } = evt { reason = Some(r); done = Some(message); }
        }
        let msg = done.expect("done");
        assert_eq!(reason, Some(StopReason::ToolUse));
        assert!(msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { id, name, arguments, .. } if id == "tc1" && name == "search" && arguments["q"] == "rust")));
    }

    #[test]
    fn test_google_foreign_thinking_downgraded_and_signatures_dropped() {
        use crate::provider::google::build_google_payload_public;
        let model = Model { id: "gemini-2.5-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        // Assistant message from a DIFFERENT model/provider -> thinking becomes plain
        // text and thought signatures are not replayed.
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking { thinking: "reasoning".into(), thinking_signature: Some("QUJD".into()), redacted: false },
                    ContentBlock::Text { text: "answer".into(), text_signature: Some("QUJD".into()) },
                ],
                timestamp: 0,
                api: Some("anthropic-messages".into()), provider: Some("anthropic".into()), model: Some("claude".into()),
                response_id: None, response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = build_google_payload_public(&model, &ctx, &StreamOptions::default());
        let parts = payload["contents"][0]["parts"].as_array().unwrap();
        // Thinking downgraded to plain text (no `thought` flag), no signatures anywhere.
        assert!(parts[0].get("thought").is_none());
        assert_eq!(parts[0]["text"], "reasoning");
        assert!(parts[0].get("thoughtSignature").is_none());
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn test_google_same_model_replays_signature_and_thought() {
        use crate::provider::google::build_google_payload_public;
        let model = Model { id: "gemini-2.5-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Thinking { thinking: "r".into(), thinking_signature: Some("QUJD".into()), redacted: false }],
                timestamp: 0,
                api: Some("google-generative-ai".into()), provider: Some("google".into()), model: Some("gemini-2.5-pro".into()),
                response_id: None, response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::Stop), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = build_google_payload_public(&model, &ctx, &StreamOptions::default());
        let part = &payload["contents"][0]["parts"][0];
        assert_eq!(part["thought"], true);
        assert_eq!(part["thoughtSignature"], "QUJD");
    }

    #[test]
    fn test_google_disables_thinking_when_no_reasoning() {
        use crate::provider::google::build_google_payload_public;
        // Gemini 2.x -> thinkingBudget 0
        let m1 = Model { id: "gemini-2.5-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let ctx = test_context();
        let p1 = build_google_payload_public(&m1, &ctx, &StreamOptions::default());
        assert_eq!(p1["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0);
        // Gemini 3 pro -> thinkingLevel LOW
        let m2 = Model { id: "gemini-3-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let p2 = build_google_payload_public(&m2, &ctx, &StreamOptions::default());
        assert_eq!(p2["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW");
    }

    #[test]
    fn test_google_gemini3_uses_thinking_level() {
        use crate::provider::google::build_google_payload_public;
        let model = Model { id: "gemini-3-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::Low), ..Default::default() };
        let payload = build_google_payload_public(&model, &ctx, &opts);
        let tc = &payload["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingLevel"], "LOW");
        assert!(tc.get("thinkingBudget").is_none());
    }

    #[test]
    fn test_google_older_model_uses_thinking_budget() {
        use crate::provider::google::build_google_payload_public;
        let model = Model { id: "gemini-2.5-pro".into(), reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let ctx = test_context();
        let opts = StreamOptions {
            reasoning: Some(ThinkingLevel::High),
            thinking_budgets: Some(ThinkingBudgets { high: Some(4096), ..Default::default() }),
            ..Default::default()
        };
        let payload = build_google_payload_public(&model, &ctx, &opts);
        let tc = &payload["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingBudget"], 4096);
        assert!(tc.get("thinkingLevel").is_none());
    }

    #[test]
    fn test_google_thinking_and_tool_config() {
        use crate::provider::google::build_google_payload_public;
        let model = Model { reasoning: true, ..test_model("google-generative-ai", "google", "https://example.com") };
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool { name: "t".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) }],
        };
        let opts = StreamOptions {
            reasoning: Some(ThinkingLevel::High),
            thinking_budgets: Some(ThinkingBudgets { high: Some(2048), ..Default::default() }),
            tool_choice: Some(serde_json::json!("required")),
            ..Default::default()
        };
        let payload = build_google_payload_public(&model, &ctx, &opts);
        assert_eq!(payload["generationConfig"]["thinkingConfig"]["includeThoughts"], true);
        assert_eq!(payload["generationConfig"]["thinkingConfig"]["thinkingBudget"], 2048);
        assert_eq!(payload["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
    }

    #[test]
    fn test_google_tool_result_function_response_merging() {
        use crate::provider::google::build_google_payload_public;
        let model = test_model("google-generative-ai", "google", "https://example.com");
        let tr = |name: &str, txt: &str, err: bool| Message {
            role: Role::ToolResult,
            content: vec![ContentBlock::Text { text: txt.into(), text_signature: None }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None, diagnostics: Vec::new(), usage: None,
            stop_reason: None, error_message: None,
            tool_call_id: Some("x".into()), tool_name: Some(name.into()),
            is_error: err, details: None,
        };
        let ctx = Context {
            system_prompt: None,
            messages: vec![tr("search", "ok", false), tr("calc", "bad", true)],
            tools: vec![],
        };
        let payload = build_google_payload_public(&model, &ctx, &StreamOptions::default());
        let contents = payload["contents"].as_array().unwrap();
        // Both function responses merge into a single user turn.
        assert_eq!(contents.len(), 1);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["functionResponse"]["name"], "search");
        assert_eq!(parts[0]["functionResponse"]["response"]["output"], "ok");
        assert_eq!(parts[1]["functionResponse"]["response"]["error"], "bad");
    }

    #[tokio::test]
    async fn test_google_in_band_error_emits_error() {
        use crate::provider::google::stream_google;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"error\":{\"code\":429,\"message\":\"quota exceeded\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);
        let mut err = None;
        while let Some(evt) = stream.next().await {
            if let Event::Error { error, .. } = evt { err = Some(error.to_string()); }
        }
        assert!(err.unwrap().contains("quota exceeded"));
    }

    #[tokio::test]
    async fn test_google_safety_finish_reason_is_error() {
        use crate::provider::google::stream_google;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"SAFETY\"}]}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);
        let mut reason = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { reason: r, .. } = evt { reason = Some(r); }
        }
        assert_eq!(reason, Some(StopReason::Error));
    }

    #[tokio::test]
    async fn test_google_stream_function_call() {
        use crate::provider::google::stream_google;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"search\",\"args\":{\"q\":\"rust\"}}}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":3,\"totalTokenCount\":8}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);

        let mut saw_tool_end = false;
        let mut done: Option<Message> = None;
        let mut reason = None;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::ToolCallEnd { name, arguments, .. } => {
                    assert_eq!(name, "search");
                    assert_eq!(arguments["q"], "rust");
                    saw_tool_end = true;
                }
                Event::Done { reason: r, message } => { reason = Some(r); done = Some(message); }
                _ => {}
            }
        }
        let msg = done.expect("done");
        assert!(saw_tool_end);
        assert_eq!(reason, Some(StopReason::ToolUse));
        assert!(msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { name, .. } if name == "search")));
    }

    #[tokio::test]
    async fn test_google_function_call_preserves_provided_id() {
        use crate::provider::google::stream_google;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"toolu_abc\",\"name\":\"search\",\"args\":{\"q\":\"r\"}}}]},\"finishReason\":\"STOP\"}]}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);
        let mut id = None;
        while let Some(evt) = stream.next().await {
            if let Event::ToolCallEnd { id: i, .. } = evt { id = Some(i); }
        }
        // The provider-supplied id is preserved (needed for tool-result pairing).
        assert_eq!(id.as_deref(), Some("toolu_abc"));
    }

    #[tokio::test]
    async fn test_google_usage_cache_and_thoughts() {
        use crate::provider::google::stream_google;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":100,\"candidatesTokenCount\":20,\"thoughtsTokenCount\":30,\"cachedContentTokenCount\":40,\"totalTokenCount\":150}}\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;
        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);
        let mut usage = None;
        while let Some(evt) = stream.next().await {
            if let Event::Done { message, .. } = evt { usage = message.usage; }
        }
        let u = usage.expect("usage");
        assert_eq!(u.input, 60); // 100 - 40 cached
        assert_eq!(u.output, 50); // 20 candidates + 30 thoughts
        assert_eq!(u.cache_read, 40);
        assert_eq!(u.total_tokens, 150);
    }

    #[test]
    fn test_mistral_reasoning_prompt_mode_and_effort() {
        use crate::provider::mistral::build_mistral_payload;
        // prompt_mode model
        let m1 = Model { id: "magistral-medium".into(), reasoning: true, ..test_model("mistral-conversations", "mistral", "https://example.com") };
        let ctx = test_context();
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let p1 = build_mistral_payload(&m1, &ctx, &opts);
        assert_eq!(p1["prompt_mode"], "reasoning");
        assert!(p1.get("reasoning_effort").is_none());

        // reasoning_effort model
        let m2 = Model { id: "mistral-medium-3.5".into(), reasoning: true, ..test_model("mistral-conversations", "mistral", "https://example.com") };
        let p2 = build_mistral_payload(&m2, &ctx, &opts);
        assert_eq!(p2["reasoning_effort"], "high");
        assert!(p2.get("prompt_mode").is_none());
    }

    #[test]
    fn test_mistral_payload_serializes_tool_history() {
        use crate::provider::mistral::build_mistral_payload;
        let model = test_model("mistral-conversations", "mistral", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolCall {
                        id: "tc1".into(), name: "search".into(),
                        arguments: std::collections::HashMap::from([("q".into(), serde_json::json!("r"))]),
                        thought_signature: None,
                    }],
                    timestamp: 0,
                    api: None, provider: None, model: None, response_id: None,
                    response_model: None, diagnostics: Vec::new(), usage: None,
                    stop_reason: Some(StopReason::ToolUse), error_message: None,
                    tool_call_id: None, tool_name: None, is_error: false, details: None,
                },
                Message {
                    role: Role::ToolResult,
                    content: vec![ContentBlock::Text { text: "found".into(), text_signature: None }],
                    timestamp: 0,
                    api: None, provider: None, model: None, response_id: None,
                    response_model: None, diagnostics: Vec::new(), usage: None,
                    stop_reason: None, error_message: None,
                    tool_call_id: Some("tc1".into()), tool_name: Some("search".into()),
                    is_error: false, details: None,
                },
            ],
            tools: vec![],
        };
        let payload = build_mistral_payload(&model, &ctx, &StreamOptions::default());
        let msgs = payload["messages"].as_array().unwrap();
        // "tc1" is normalized to a 9-char alphanumeric id, consistent across messages.
        let norm_id = msgs[0]["tool_calls"][0]["id"].as_str().unwrap().to_string();
        assert_eq!(norm_id.len(), 9);
        assert!(norm_id.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], "search");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], norm_id);
        assert_eq!(msgs[1]["content"], "found");
    }

    #[test]
    fn test_mistral_tool_id_passthrough_when_already_valid() {
        use crate::provider::mistral::build_mistral_payload;
        let model = test_model("mistral-conversations", "mistral", "https://example.com");
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall {
                    id: "abc123XYZ".into(), name: "t".into(),
                    arguments: std::collections::HashMap::new(), thought_signature: None,
                }],
                timestamp: 0,
                api: None, provider: None, model: None, response_id: None,
                response_model: None, diagnostics: Vec::new(), usage: None,
                stop_reason: Some(StopReason::ToolUse), error_message: None,
                tool_call_id: None, tool_name: None, is_error: false, details: None,
            }],
            tools: vec![],
        };
        let payload = build_mistral_payload(&model, &ctx, &StreamOptions::default());
        // A 9-char alphanumeric id is preserved as-is.
        assert_eq!(payload["messages"][0]["tool_calls"][0]["id"], "abc123XYZ");
    }
}
