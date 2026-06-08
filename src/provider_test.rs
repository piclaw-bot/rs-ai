//! Provider-level tests using mock HTTP servers.

#[cfg(test)]
mod tests {
    use crate::provider::openai::stream_openai;
    use crate::provider::anthropic::stream_anthropic;
    use crate::provider::mistral::stream_mistral;
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

        while let Some(evt) = stream.next().await {
            match evt {
                Event::ToolCallStart { id, name } => {
                    assert_eq!(id, "tc1");
                    assert_eq!(name, "search");
                    saw_tool_start = true;
                }
                Event::ToolCallDelta { delta } => tool_json.push_str(&delta),
                Event::Done { reason, .. } => stop_reason = Some(reason),
                _ => {}
            }
        }

        assert!(saw_tool_start);
        assert_eq!(tool_json, "{\"q\":\"rust\"}");
        assert_eq!(stop_reason, Some(StopReason::ToolUse));
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
}
