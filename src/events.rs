//! Stream event types emitted by providers.

use std::sync::Arc;

use crate::types::{Message, StopReason};

/// Events emitted during streaming.
#[derive(Debug, Clone)]
pub enum Event {
    /// Stream has started.
    Start { partial: Message },
    /// Incremental text content.
    TextDelta { delta: String },
    /// Text block started.
    TextStart,
    /// Text block ended.
    TextEnd,
    /// Incremental thinking/reasoning content.
    ThinkingDelta { delta: String },
    /// Thinking block started.
    ThinkingStart,
    /// Thinking block ended.
    ThinkingEnd,
    /// Tool call started.
    ToolCallStart { id: String, name: String },
    /// Incremental tool call argument JSON.
    ToolCallDelta { delta: String },
    /// Tool call ended with parsed arguments.
    ToolCallEnd {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Stream completed successfully.
    Done {
        reason: StopReason,
        message: Message,
    },
    /// Stream failed.
    Error {
        reason: StopReason,
        error: Arc<dyn std::error::Error + Send + Sync>,
        message: Option<Message>,
    },
}
