//! Faux (test double) provider for unit testing without network calls.

use std::sync::Arc;

use crate::events::Event;
use crate::types::*;

/// Create a faux stream that emits a single text response.
pub fn stream_faux_text<'a>(
    text: &'a str,
    model: &'a Model,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    let text = text.to_string();
    let model_clone = model.clone();
    Box::pin(async_stream::stream! {
        let partial = Message {
            role: Role::Assistant,
            content: Vec::new(),
            timestamp: 0,
            api: Some(model_clone.api.clone()),
            provider: Some(model_clone.provider.clone()),
            model: Some(model_clone.id.clone()),
            response_id: Some("faux-response".into()),
            response_model: None,
            diagnostics: Vec::new(),
            usage: Some(Usage {
                input: 10,
                output: text.len() as u32 / 4,
                total_tokens: 10 + text.len() as u32 / 4,
                ..Default::default()
            }),
            stop_reason: None,
            error_message: None,
            tool_call_id: None,
            tool_name: None,
            is_error: false,
            details: None,
        };
        yield Event::Start { partial: partial.clone() };
        yield Event::TextStart;

        // Emit in chunks
        for chunk in text.as_bytes().chunks(20) {
            let s = String::from_utf8_lossy(chunk).to_string();
            yield Event::TextDelta { delta: s };
        }

        yield Event::TextEnd;

        let final_msg = Message {
            content: vec![ContentBlock::Text { text: text.clone(), text_signature: None }],
            stop_reason: Some(StopReason::Stop),
            usage: partial.usage.clone(),
            ..partial
        };
        yield Event::Done { reason: StopReason::Stop, message: final_msg };
    })
}

/// Create a faux stream that immediately errors.
pub fn stream_faux_error<'a>(
    error_msg: &'a str,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    let msg = error_msg.to_string();
    Box::pin(async_stream::stream! {
        yield Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
            message: None,
        };
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

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
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: None,
            compat: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_faux_text_stream() {
        let model = faux_model();
        let mut stream = stream_faux_text("Hello, world!", &model);
        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt);
        }
        // Start, TextStart, TextDelta, TextEnd, Done
        assert!(events.len() >= 4);
        assert!(matches!(&events[0], Event::Start { .. }));
        assert!(matches!(&events[1], Event::TextStart));
        assert!(matches!(events.last().unwrap(), Event::Done { .. }));
    }

    #[tokio::test]
    async fn test_faux_error_stream() {
        let mut stream = stream_faux_error("test failure");
        let evt = stream.next().await.unwrap();
        assert!(matches!(evt, Event::Error { .. }));
    }
}
