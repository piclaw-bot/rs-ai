//! Context overflow detection.
//!
//! Determines whether a response indicates the model ran out of context window.

use crate::types::{Message, StopReason, Model};

/// Check if a message indicates context overflow.
pub fn is_context_overflow(msg: &Message, model: &Model) -> bool {
    // Check stop reason
    if msg.stop_reason == Some(StopReason::Length) {
        return true;
    }

    // Check if error message mentions context/token limits
    if let Some(ref err) = msg.error_message {
        let lower = err.to_lowercase();
        if lower.contains("context") && (lower.contains("length") || lower.contains("exceed") || lower.contains("limit"))
            || lower.contains("maximum context")
            || lower.contains("token limit")
            || lower.contains("too many tokens")
            || lower.contains("reduce the length")
            || lower.contains("context_length_exceeded")
        {
            return true;
        }
    }

    // Check usage vs context window
    if let Some(ref usage) = msg.usage
        && model.context_window > 0 && usage.input + usage.output >= model.context_window {
            return true;
        }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, Role, ContentBlock};

    fn test_model() -> Model {
        Model {
            id: "test".into(),
            name: "Test".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 4096,
            max_tokens: 1024,
            headers: None,
            api_key: None,
        }
    }

    fn base_msg() -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: "hi".into(), text_signature: None }],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }
    }

    #[test]
    fn test_stop_reason_length() {
        let mut msg = base_msg();
        msg.stop_reason = Some(StopReason::Length);
        assert!(is_context_overflow(&msg, &test_model()));
    }

    #[test]
    fn test_error_message_overflow() {
        let mut msg = base_msg();
        msg.error_message = Some("context_length_exceeded".into());
        assert!(is_context_overflow(&msg, &test_model()));
    }

    #[test]
    fn test_normal_stop() {
        let mut msg = base_msg();
        msg.stop_reason = Some(StopReason::Stop);
        assert!(!is_context_overflow(&msg, &test_model()));
    }
}
