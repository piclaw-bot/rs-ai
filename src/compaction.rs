//! Context compaction — summarize long conversations to fit context windows.

use crate::types::{Context, ContentBlock, Message, Role};

/// Compact a context by removing older messages when approaching context limits.
///
/// Keeps the system prompt, the most recent `keep_recent` messages, and
/// replaces everything in between with a summary placeholder.
pub fn compact_context(ctx: &Context, keep_recent: usize, summary: Option<&str>) -> Context {
    if ctx.messages.len() <= keep_recent {
        return ctx.clone();
    }

    let mut messages = Vec::new();

    // Add summary of removed messages
    if let Some(text) = summary {
        messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: format!("[Previous conversation summarized: {}]", text),
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
            tool_call_id: None,
            tool_name: None,
            is_error: false,
        });
    }

    // Keep the most recent messages
    let start = ctx.messages.len().saturating_sub(keep_recent);
    messages.extend_from_slice(&ctx.messages[start..]);

    Context {
        system_prompt: ctx.system_prompt.clone(),
        messages,
        tools: ctx.tools.clone(),
    }
}

/// Estimate token count for a context (rough approximation: 4 chars ≈ 1 token).
pub fn estimate_tokens(ctx: &Context) -> u32 {
    let mut chars = 0u32;
    if let Some(ref prompt) = ctx.system_prompt {
        chars += prompt.len() as u32;
    }
    for msg in &ctx.messages {
        for block in &msg.content {
            match block {
                ContentBlock::Text { text, .. } => chars += text.len() as u32,
                ContentBlock::Thinking { thinking, .. } => chars += thinking.len() as u32,
                ContentBlock::Image { data, .. } => chars += data.len() as u32 / 4, // base64 overhead
                ContentBlock::ToolCall { arguments, .. } => {
                    chars += serde_json::to_string(arguments).unwrap_or_default().len() as u32;
                }
            }
        }
    }
    chars / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::user_message;

    #[test]
    fn test_compact_no_op_when_short() {
        let ctx = Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![user_message("hi"), user_message("there")],
            tools: vec![],
        };
        let result = compact_context(&ctx, 5, None);
        assert_eq!(result.messages.len(), 2);
    }

    #[test]
    fn test_compact_keeps_recent() {
        let ctx = Context {
            system_prompt: None,
            messages: (0..20).map(|i| user_message(&format!("msg {}", i))).collect(),
            tools: vec![],
        };
        let result = compact_context(&ctx, 3, Some("first 17 messages about greetings"));
        assert_eq!(result.messages.len(), 4); // summary + 3 recent
    }

    #[test]
    fn test_estimate_tokens() {
        let ctx = Context {
            system_prompt: Some("System prompt here.".into()),
            messages: vec![user_message("Hello world, this is a test message.")],
            tools: vec![],
        };
        let tokens = estimate_tokens(&ctx);
        assert!(tokens > 10);
    }
}
