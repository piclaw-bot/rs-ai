//! OpenAI Codex Responses provider (WebSocket + SSE fallback) — stub.
//!
//! Full implementation requires WebSocket support via tokio-tungstenite;
//! this stub defines the entry point.

use std::sync::Arc;
use crate::events::Event;
use crate::types::*;

/// Start a Codex stream (stub — returns error until WebSocket transport implemented).
pub fn stream_codex<'a>(
    model: &'a Model,
    _context: &'a Context,
    _opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    Box::pin(async_stream::stream! {
        yield Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                format!("codex provider not yet implemented for model: {}", model.id),
            )),
            message: None,
        };
    })
}
