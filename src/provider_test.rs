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
        assert_eq!(payload["session_id"], "sess-1");
        assert_eq!(payload["prompt_cache_key"], "sess-1");
        assert_eq!(payload["prompt_cache_retention"], "24h");
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
}
