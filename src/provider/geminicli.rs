//! Google Gemini CLI (Cloud Code Assist) provider — stub.

use std::sync::Arc;
use crate::events::Event;
use crate::types::*;

/// Start a Gemini CLI stream (stub).
pub fn stream_geminicli<'a>(
    model: &'a Model,
    _context: &'a Context,
    _opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    Box::pin(async_stream::stream! {
        yield Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                format!("geminicli provider not yet implemented for model: {}", model.id),
            )),
            message: None,
        };
    })
}
