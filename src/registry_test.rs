#[cfg(test)]
mod tests {
    use crate::registry;
    use crate::types::provider_id;

    #[test]
    fn test_register_builtin_models() {
        registry::register_builtin_models();
        let providers = registry::list_providers();
        assert!(providers.len() >= 30, "expected 30+ providers, got {}", providers.len());
        
        let model = registry::get_model(provider_id::OPENAI, "gpt-4o");
        assert!(model.is_some(), "gpt-4o should be registered");
        let m = model.unwrap();
        assert!(!m.api.is_empty());
        assert!(m.context_window > 0);
    }

    #[test]
    fn test_get_model_not_found() {
        registry::register_builtin_models();
        assert!(registry::get_model("nonexistent", "fake").is_none());
    }

    #[test]
    fn test_list_models_filter() {
        registry::register_builtin_models();
        let openai = registry::list_models(Some(provider_id::OPENAI));
        assert!(openai.len() >= 10, "expected 10+ OpenAI models, got {}", openai.len());
        for m in &openai {
            assert_eq!(m.provider, provider_id::OPENAI);
        }
    }
}
