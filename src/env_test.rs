#[cfg(test)]
mod tests {
    use crate::env::{get_env_api_key, resolve_api_key};
    use crate::types::{Model, ModelCost, StreamOptions};

    fn test_model(provider: &str) -> Model {
        Model {
            id: "test".into(),
            name: "Test".into(),
            api: "openai-completions".into(),
            provider: provider.into(),
            base_url: "https://example.com".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: None,
        }
    }

    #[test]
    fn test_resolve_api_key_from_opts() {
        let model = test_model("openai");
        let opts = StreamOptions {
            api_key: Some("sk-test".into()),
            ..Default::default()
        };
        assert_eq!(resolve_api_key(&model, &opts), Some("sk-test".into()));
    }

    #[test]
    fn test_resolve_api_key_from_model() {
        let mut model = test_model("openai");
        model.api_key = Some("sk-model".into());
        let opts = StreamOptions::default();
        assert_eq!(resolve_api_key(&model, &opts), Some("sk-model".into()));
    }

    #[test]
    fn test_env_fallback_generic() {
        unsafe { std::env::set_var("TOTALLY_CUSTOM_PROVIDER_API_KEY", "custom-key"); }
        let key = get_env_api_key("totally-custom-provider");
        assert_eq!(key, Some("custom-key".into()));
        unsafe { std::env::remove_var("TOTALLY_CUSTOM_PROVIDER_API_KEY"); }
    }
}
