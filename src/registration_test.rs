#[cfg(test)]
mod tests {
    use crate::provider;
    use crate::registry;
    use crate::types::*;
    use crate::images;

    fn openai_model() -> Model {
        Model {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "http://localhost:1".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: Some("test-key".into()),
        }
    }

    #[test]
    fn test_register_builtin_providers() {
        registry::clear_api_providers();
        provider::register_builtin_providers();
        assert!(registry::get_api_provider("openai-completions"));
        assert!(registry::get_api_provider("openai-responses"));
        assert!(registry::get_api_provider("anthropic-messages"));
        assert!(registry::get_api_provider("google-generative-ai"));
        assert!(registry::get_api_provider("mistral-conversations"));
        assert!(registry::get_api_provider("bedrock-converse-stream"));
        assert!(registry::get_api_provider("openai-codex-responses"));
        assert!(registry::get_api_provider("google-gemini-cli"));
    }

    #[tokio::test]
    async fn test_complete_uses_registered_provider() {
        registry::clear_api_providers();
        provider::register_builtin_providers();
        let model = openai_model();
        let ctx = Context { system_prompt: None, messages: vec![user_message("hi")], tools: vec![] };
        // base_url points nowhere, so once dispatch works we should get a network error, not the old placeholder error
        let err = registry::complete(&model, &ctx, &StreamOptions::default()).await.unwrap_err();
        assert!(!err.to_string().contains("provider stream not yet implemented"));
    }

    #[test]
    fn test_register_builtin_image_models() {
        images::clear_image_models();
        images::register_builtin_image_models();
        let providers = images::list_image_providers();
        assert!(providers.contains(&"openrouter".to_string()));
        let models = images::list_image_models(Some("openrouter"));
        assert!(models.len() >= 30);
        let model = images::get_image_model("openrouter", &models[0].id);
        assert!(model.is_some());
    }

    #[test]
    fn test_register_builtin_image_providers() {
        images::clear_images_api_providers();
        images::register_builtin_image_providers();
        let model = images::ImagesModel {
            id: "dummy".into(),
            name: "Dummy".into(),
            api: "openrouter-images".into(),
            provider: "openrouter".into(),
            base_url: "http://localhost:1".into(),
            input: vec!["text".into()],
            output: vec!["image".into()],
            cost: ModelCost::default(),
        };
        // generate_images should find a provider and not hit the missing-provider fallback
        let ctx = images::ImagesContext { input: vec![] };
        let opts = images::openrouter::ImagesOptions::default();
        let fut = images::generate_images(&model, &ctx, &opts);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(fut);
        assert!(!matches!(result.error_message.as_deref(), Some(msg) if msg.contains("no image provider registered")));
    }
}
