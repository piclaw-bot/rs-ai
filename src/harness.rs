//! Harness / agent helpers for building LLM agent loops.
//!
//! Provides context cloning, message appending, text extraction, and
//! conversation management utilities.

use crate::types::{ContentBlock, Context, Message, Role, StopReason};

/// Extract the full text content from a message.
pub fn get_text_content(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Append an assistant message to a context (consumes and returns).
pub fn append_assistant_message(mut ctx: Context, msg: &Message) -> Context {
    ctx.messages.push(msg.clone());
    ctx
}

/// Create a user message and append it to context.
pub fn append_user_message(mut ctx: Context, text: &str) -> Context {
    ctx.messages.push(crate::types::user_message(text));
    ctx
}

/// Deep clone a context.
pub fn clone_context(ctx: &Context) -> Context {
    ctx.clone()
}

/// Check if a message indicates the model wants to use tools.
pub fn is_tool_use(msg: &Message) -> bool {
    msg.stop_reason == Some(StopReason::ToolUse)
        || msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. }))
}

/// Extract tool calls from a message.
pub fn get_tool_calls(msg: &Message) -> Vec<&ContentBlock> {
    msg.content
        .iter()
        .filter(|b| matches!(b, ContentBlock::ToolCall { .. }))
        .collect()
}

/// Create a tool result message.
pub fn tool_result_message(tool_call_id: &str, tool_name: &str, result: &str, is_error: bool) -> Message {
    Message {
        role: Role::ToolResult,
        content: vec![ContentBlock::Text {
            text: result.to_string(),
            text_signature: None,
        }],
        timestamp: 0,
        api: None,
        provider: None,
        model: None,
        response_id: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        tool_call_id: Some(tool_call_id.to_string()),
        tool_name: Some(tool_name.to_string()),
        is_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_get_text_content() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text { text: "Hello ".into(), text_signature: None },
                ContentBlock::Text { text: "world".into(), text_signature: None },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
        };
        assert_eq!(get_text_content(&msg), "Hello world");
    }

    #[test]
    fn test_is_tool_use() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall {
                id: "tc1".into(),
                name: "search".into(),
                arguments: HashMap::new(),
                thought_signature: None,
            }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            usage: None, stop_reason: Some(StopReason::ToolUse), error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
        };
        assert!(is_tool_use(&msg));
    }

    #[test]
    fn test_tool_result_message() {
        let msg = tool_result_message("tc1", "search", "found it", false);
        assert_eq!(msg.role, Role::ToolResult);
        assert_eq!(msg.tool_call_id.as_deref(), Some("tc1"));
        assert!(!msg.is_error);
    }
}
