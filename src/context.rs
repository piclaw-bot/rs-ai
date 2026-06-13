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

    // Check if the error message matches any known overflow phrasing (mirrors upstream
    // overflow.js patterns across providers).
    if let Some(ref err) = msg.error_message {
        let lower = err.to_lowercase();
        const NEEDLES: &[&str] = &[
            "prompt is too long",                 // Anthropic
            "request_too_large",                  // Anthropic 413
            "request exceeds the maximum size",   // Anthropic 413
            "input is too long for requested model", // Bedrock
            "exceeds the context window",         // OpenAI
            "maximum context length",             // OpenAI / LiteLLM / OpenRouter / Mistral
            "maximum prompt length",              // xAI / Grok
            "reduce the length of the messages",  // Groq
            "maximum allowed input length",       // OpenRouter / Poolside
            "context length",                     // Together AI / generic
            "exceeds the limit of",               // GitHub Copilot
            "exceeds the available context size", // llama.cpp
            "greater than the context length",    // LM Studio
            "context window exceeds limit",       // MiniMax
            "exceeded model token limit",         // Kimi
            "too large for model",                // Mistral
            "model_context_window_exceeded",      // z.ai
            "prompt too long",                    // Ollama
            "context_length_exceeded",            // generic
            "context length exceeded",            // generic
            "token limit",
            "too many tokens",
            "input token count",                  // Google (with "exceeds the maximum")
        ];
        if NEEDLES.iter().any(|n| lower.contains(n)) {
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
    fn test_overflow_provider_phrasings() {
        let cases = [
            "prompt is too long: 213462 tokens > 200000 maximum",          // Anthropic
            "Your input exceeds the context window of this model",          // OpenAI
            "Input length (265330) exceeds model's maximum context length (262144).", // LiteLLM
            "The input token count (1196265) exceeds the maximum number of tokens allowed", // Google
            "prompt token count of 50000 exceeds the limit of 8192",        // Copilot
            "input is too long for requested model",                        // Bedrock
            "413 request_too_large",                                        // Anthropic 413
        ];
        for c in cases {
            let mut msg = base_msg();
            msg.error_message = Some(c.to_string());
            assert!(is_context_overflow(&msg, &test_model()), "should detect overflow: {c}");
        }
    }

    #[test]
    fn test_normal_stop() {
        let mut msg = base_msg();
        msg.stop_reason = Some(StopReason::Stop);
        assert!(!is_context_overflow(&msg, &test_model()));
    }
}
