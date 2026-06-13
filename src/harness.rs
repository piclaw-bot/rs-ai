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
        response_model: None,
        diagnostics: Vec::new(),
        usage: None,
        stop_reason: None,
        error_message: None,
        tool_call_id: Some(tool_call_id.to_string()),
        tool_name: Some(tool_name.to_string()),
        is_error,
        details: None,
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
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
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
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: Some(StopReason::ToolUse), error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
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

/// Append a tool result to a context.
pub fn append_tool_result(mut ctx: Context, tool_call_id: &str, tool_name: &str, result: &str, is_error: bool) -> Context {
    ctx.messages.push(tool_result_message(tool_call_id, tool_name, result, is_error));
    ctx
}

/// Check if a message has tool calls.
pub fn has_tool_calls(msg: &Message) -> bool {
    msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. }))
}

/// Check if a message needs tool execution (tool_use stop + has tool calls).
pub fn needs_tool_execution(msg: &Message) -> bool {
    is_tool_use(msg) && has_tool_calls(msg)
}

/// Check if context fits in a model's context window (rough estimate).
pub fn fits_in_context_window(ctx: &Context, model: &crate::types::Model) -> bool {
    let est = crate::compaction::estimate_tokens(ctx);
    est < model.context_window
}

/// Save context to JSON string.
pub fn save_context(ctx: &Context) -> Result<String, serde_json::Error> {
    serde_json::to_string(ctx)
}

/// Load context from JSON string.
pub fn load_context(json: &str) -> Result<Context, serde_json::Error> {
    serde_json::from_str(json)
}

/// Check if two models are the same (by provider + id).
pub fn models_are_equal(a: &crate::types::Model, b: &crate::types::Model) -> bool {
    a.provider == b.provider && a.id == b.id
}

/// Hook invoked with the serialized request payload before sending.
pub type PayloadHookFn<'a> = dyn Fn(serde_json::Value) -> serde_json::Value + 'a;
/// Hook invoked with the HTTP status and response headers.
pub type ResponseHookFn<'a> = dyn Fn(u16, &std::collections::HashMap<String, String>) + 'a;

/// Invoke an on-payload hook (placeholder for hook infrastructure).
/// In Go this mutates the payload before sending; in Rust we return the modified value.
pub fn invoke_on_payload(
    payload: serde_json::Value,
    on_payload: Option<&PayloadHookFn<'_>>,
) -> serde_json::Value {
    match on_payload {
        Some(f) => f(payload),
        None => payload,
    }
}

/// Invoke an on-response hook (placeholder for hook infrastructure).
pub fn invoke_on_response(
    status: u16,
    headers: &std::collections::HashMap<String, String>,
    on_response: Option<&ResponseHookFn<'_>>,
) {
    if let Some(f) = on_response {
        f(status, headers);
    }
}
