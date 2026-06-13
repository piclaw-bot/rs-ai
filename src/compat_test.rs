#[cfg(test)]
mod tests {
    use crate::compat::*;
    use crate::types::{Model, ModelCost};

    fn model_with(provider: &str, base_url: &str, id: &str) -> Model {
        Model {
            id: id.into(),
            name: "Test".into(),
            api: "openai-completions".into(),
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: None,
        }
    }

    #[test]
    fn test_openai_defaults() {
        let m = model_with("openai", "https://api.openai.com/v1", "gpt-4o");
        let c = detect_compat(&m);
        assert_eq!(c.supports_developer_role, Some(true));
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_completion_tokens"));
    }

    #[test]
    fn test_ollama_detection() {
        let m = model_with("ollama", "http://localhost:11434/v1", "llama3");
        let c = detect_compat(&m);
        assert_eq!(c.supports_strict_mode, Some(false));
        assert_eq!(c.requires_tool_result_name, Some(true));
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_remote_11434_not_ollama() {
        let m = model_with("custom", "https://example.com:11434/v1", "model");
        let c = detect_compat(&m);
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_completion_tokens"));
    }

    #[test]
    fn test_openrouter_developer_role() {
        let m = model_with("openrouter", "https://openrouter.ai/api/v1", "meta/llama");
        let c = detect_compat(&m);
        assert_eq!(c.supports_developer_role, Some(false)); // non-anthropic/openai prefix

        let m2 = model_with("openrouter", "https://openrouter.ai/api/v1", "anthropic/claude");
        let c2 = detect_compat(&m2);
        assert_eq!(c2.supports_developer_role, Some(true)); // anthropic prefix
    }

    #[test]
    fn test_deepseek_thinking() {
        let m = model_with("deepseek", "https://api.deepseek.com/v1", "deepseek-v4");
        let c = detect_compat(&m);
        assert_eq!(c.thinking_format.as_deref(), Some("deepseek"));
        assert_eq!(c.requires_reasoning_content_on_assistant_messages, Some(true));
    }

    #[test]
    fn test_xiaomi_as_deepseek() {
        let m = model_with("xiaomi", "https://api.xiaomimimo.com/v1", "mimo");
        let c = detect_compat(&m);
        assert_eq!(c.thinking_format.as_deref(), Some("deepseek"));
    }

    #[test]
    fn test_ant_ling() {
        let m = model_with("ant-ling", "https://api.ant-ling.com/v1", "ling");
        let c = detect_compat(&m);
        assert_eq!(c.thinking_format.as_deref(), Some("ant-ling"));
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_nvidia() {
        let m = model_with("nvidia", "https://integrate.api.nvidia.com/v1", "nim");
        let c = detect_compat(&m);
        assert_eq!(c.supports_store, Some(false));
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_tokens"));
        assert_eq!(c.supports_reasoning_effort, Some(false));
        assert_eq!(c.supports_strict_mode, Some(false));
        assert_eq!(c.supports_long_cache_retention, Some(false));
    }

    #[test]
    fn test_grok_no_reasoning_effort() {
        let m = model_with("xai", "https://api.x.ai/v1", "grok-2");
        let c = detect_compat(&m);
        assert_eq!(c.supports_reasoning_effort, Some(false));
    }

    #[test]
    fn test_together_detection() {
        let m = model_with("together", "https://api.together.ai/v1", "deepseek-ai/DeepSeek-R1");
        let c = detect_compat(&m);
        assert_eq!(c.thinking_format.as_deref(), Some("together"));
        assert_eq!(c.supports_reasoning_effort, Some(false));
        assert_eq!(c.supports_strict_mode, Some(false));
        assert_eq!(c.supports_long_cache_retention, Some(false));
        assert_eq!(c.max_tokens_field.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_compat_merge() {
        let m = model_with("openai", "https://api.openai.com/v1", "gpt-4o");
        let overrides = OpenAICompletionsCompat {
            supports_temperature: Some(false),
            ..Default::default()
        };
        let c = detect_compat_for_model(&m, Some(&overrides));
        assert_eq!(c.supports_temperature, Some(false));
        assert_eq!(c.supports_developer_role, Some(true)); // base preserved
    }
}
