//! Context overflow detection.
//!
//! Determines whether a response indicates the model ran out of context window.

use crate::types::{Message, StopReason, Model};

/// Check if a message indicates context overflow (mirrors upstream isContextOverflow).
pub fn is_context_overflow(msg: &Message, model: &Model) -> bool {
    let ctx = model.context_window;

    // Case 1: error-message patterns (only for error stops), excluding known
    // non-overflow errors such as throttling / rate limiting.
    if msg.stop_reason == Some(StopReason::Error)
        && let Some(ref err) = msg.error_message {
        let lower = err.to_lowercase();
        const NON_OVERFLOW: &[&str] = &[
            "throttling error",   // AWS Bedrock formatted throttling
            "service unavailable",
            "rate limit",
            "too many requests",
        ];
        const OVERFLOW: &[&str] = &[
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
            "token limit exceeded",               // generic
            "too many tokens",                    // generic
            "input token count",                  // Google (with "exceeds the maximum")
        ];
        let is_non_overflow = NON_OVERFLOW.iter().any(|n| lower.contains(n));
        if !is_non_overflow && OVERFLOW.iter().any(|n| lower.contains(n)) {
            return true;
        }
    }

    if let Some(ref usage) = msg.usage {
        let input_tokens = usage.input + usage.cache_read;
        // Case 2: silent overflow (z.ai style) - successful but input exceeds context.
        if ctx > 0 && msg.stop_reason == Some(StopReason::Stop) && input_tokens > ctx {
            return true;
        }
        // Case 3: length-stop overflow (Xiaomi MiMo style) - server truncated oversized
        // input to fill the context window, leaving no room for output.
        if ctx > 0
            && msg.stop_reason == Some(StopReason::Length)
            && usage.output == 0
            && f64::from(input_tokens) >= f64::from(ctx) * 0.99 {
            return true;
        }
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
            compat: Default::default(),
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
        // A normal max-tokens truncation (output > 0) is NOT context overflow.
        let mut msg = base_msg();
        msg.stop_reason = Some(StopReason::Length);
        msg.usage = Some(crate::types::Usage { input: 10, output: 50, ..Default::default() });
        assert!(!is_context_overflow(&msg, &test_model()));
        // Length stop with output==0 and input filling the window IS overflow (MiMo style).
        let mut msg2 = base_msg();
        msg2.stop_reason = Some(StopReason::Length);
        msg2.usage = Some(crate::types::Usage { input: 4096, output: 0, ..Default::default() });
        assert!(is_context_overflow(&msg2, &test_model()));
    }

    #[test]
    fn test_error_message_overflow() {
        let mut msg = base_msg();
        msg.stop_reason = Some(StopReason::Error);
        msg.error_message = Some("context_length_exceeded".into());
        assert!(is_context_overflow(&msg, &test_model()));
        // Throttling that merely mentions tokens is excluded (NON_OVERFLOW).
        let mut throttle = base_msg();
        throttle.stop_reason = Some(StopReason::Error);
        throttle.error_message = Some("Throttling error: Too many tokens, please wait".into());
        assert!(!is_context_overflow(&throttle, &test_model()));
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
            msg.stop_reason = Some(StopReason::Error);
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
