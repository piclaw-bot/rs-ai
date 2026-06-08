//! Provider retry and additional mock tests.

#[cfg(test)]
mod tests {
    use crate::provider::openai::stream_openai;
    use crate::provider::google::stream_google;
    use crate::provider::responses::stream_responses;
    use crate::types::*;
    use crate::events::Event;
    use tokio_stream::StreamExt;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::{method, path};

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

    // --- OpenAI Retry (429) ---

    #[tokio::test]
    async fn test_openai_429_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("Rate limited"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut saw_error = false;
        while let Some(evt) = stream.next().await {
            if let Event::Error { error, .. } = evt {
                assert!(error.to_string().contains("429"));
                saw_error = true;
            }
        }
        assert!(saw_error);
    }

    // --- Google Generative AI ---

    #[tokio::test]
    async fn test_google_stream_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello from Gemini\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":4,\"totalTokenCount\":9}}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);

        let mut text = String::new();
        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::TextDelta { delta } => text.push_str(&delta),
                Event::Done { message, .. } => {
                    assert!(message.usage.is_some());
                    assert_eq!(message.usage.as_ref().unwrap().input, 5);
                    saw_done = true;
                }
                _ => {}
            }
        }
        assert_eq!(text, "Hello from Gemini");
        assert!(saw_done);
    }

    #[tokio::test]
    async fn test_google_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(403).set_body_string("Forbidden"))
            .mount(&server)
            .await;

        let model = test_model("google-generative-ai", "google", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);

        let evt = stream.next().await.unwrap();
        // Skip Start event if present
        let evt = if matches!(evt, Event::Start { .. }) {
            stream.next().await.unwrap()
        } else {
            evt
        };
        assert!(matches!(evt, Event::Error { .. }));
    }

    // --- OpenAI Responses ---

    #[tokio::test]
    async fn test_responses_stream_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string(
                    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-1\"}}\n\n\
                     data: {\"type\":\"response.content_part.added\"}\n\n\
                     data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello!\"}\n\n\
                     data: {\"type\":\"response.content_part.done\"}\n\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}\n\n"
                )
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-responses", "openai", &server.uri());
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);

        let mut text = String::new();
        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            match evt {
                Event::TextDelta { delta } => text.push_str(&delta),
                Event::Done { message, .. } => {
                    assert_eq!(message.response_id.as_deref(), Some("resp-1"));
                    assert_eq!(message.usage.as_ref().unwrap().input, 8);
                    saw_done = true;
                }
                _ => {}
            }
        }
        assert_eq!(text, "Hello!");
        assert!(saw_done);
    }

    // --- Missing key / error paths ---

    #[tokio::test]
    async fn test_google_missing_key() {
        let model = Model { api_key: None, ..test_model("google-generative-ai", "google", "http://x") };
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_google(&model, &ctx, &opts);
        let evt = stream.next().await.unwrap();
        assert!(matches!(evt, Event::Error { .. }));
    }

    #[tokio::test]
    async fn test_responses_missing_key() {
        let model = Model { api_key: None, ..test_model("openai-responses", "openai", "http://x") };
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_responses(&model, &ctx, &opts);
        let evt = stream.next().await.unwrap();
        assert!(matches!(evt, Event::Error { .. }));
    }

    // --- OpenAI with reasoning payload ---

    #[tokio::test]
    async fn test_openai_reasoning_payload() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string("data: {\"choices\":[{\"delta\":{\"content\":\"thought\"},\"index\":0}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = test_model("openai-completions", "openai", &server.uri());
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::Medium), ..Default::default() };
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut saw_done = false;
        while let Some(evt) = stream.next().await {
            if matches!(evt, Event::Done { .. }) { saw_done = true; }
        }
        assert!(saw_done);
    }

    // --- Cloudflare header ---

    #[tokio::test]
    async fn test_openai_cloudflare_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wiremock::matchers::header("cf-aig-authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200)
                .set_body_string("data: {\"choices\":[{\"delta\":{\"content\":\"cf\"},\"index\":0}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\ndata: [DONE]\n\n")
                .insert_header("content-type", "text/event-stream"))
            .mount(&server)
            .await;

        let model = Model {
            provider: "cloudflare-ai-gateway".into(),
            ..test_model("openai-completions", "cloudflare-ai-gateway", &server.uri())
        };
        let opts = StreamOptions::default();
        let ctx = test_context();
        let mut stream = stream_openai(&model, &ctx, &opts);

        let mut text = String::new();
        while let Some(evt) = stream.next().await {
            if let Event::TextDelta { delta } = evt { text.push_str(&delta); }
        }
        assert_eq!(text, "cf");
    }
}
