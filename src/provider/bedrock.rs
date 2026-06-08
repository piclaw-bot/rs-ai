//! Amazon Bedrock ConverseStream provider (stub).
//!
//! Full implementation requires the AWS SDK for Rust; this stub defines the
//! provider entry point and will be completed when aws-sdk-bedrockruntime
//! is added as a dependency.

use std::sync::Arc;
use crate::events::Event;
use crate::types::*;

/// Start a Bedrock ConverseStream (stub — returns error until AWS SDK integrated).
pub fn stream_bedrock<'a>(
    model: &'a Model,
    _context: &'a Context,
    _opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    Box::pin(async_stream::stream! {
        yield Event::Error {
            reason: StopReason::Error,
            error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                format!("bedrock provider not yet implemented for model: {}", model.id),
            )),
            message: None,
        };
    })
}
