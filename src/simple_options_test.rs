#[cfg(test)]
mod tests {
    use crate::simple_options::*;
    use crate::types::{Model, ModelCost, ThinkingLevel, ModelThinkingLevel};
    use std::collections::HashMap;

    fn reasoning_model(map: Option<HashMap<String, Option<String>>>) -> Model {
        Model {
            id: "test".into(),
            name: "Test".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "".into(),
            reasoning: true,
            thinking_level_map: map,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 16384,
            headers: None,
            api_key: None,
        }
    }

    #[test]
    fn test_supported_levels_default() {
        let model = reasoning_model(None);
        let levels = get_supported_thinking_levels(&model);
        assert!(levels.contains(&ModelThinkingLevel::Off));
        assert!(levels.contains(&ModelThinkingLevel::Medium));
        assert!(!levels.contains(&ModelThinkingLevel::XHigh)); // xhigh must be explicit
    }

    #[test]
    fn test_supported_levels_with_map() {
        let map = HashMap::from([
            ("off".into(), None), // disabled
            ("low".into(), None), // disabled
            ("high".into(), Some("high".into())),
            ("xhigh".into(), Some("max".into())),
        ]);
        let model = reasoning_model(Some(map));
        let levels = get_supported_thinking_levels(&model);
        assert!(!levels.contains(&ModelThinkingLevel::Off));
        assert!(!levels.contains(&ModelThinkingLevel::Low));
        assert!(levels.contains(&ModelThinkingLevel::High));
        assert!(levels.contains(&ModelThinkingLevel::XHigh));
        assert!(levels.contains(&ModelThinkingLevel::Minimal)); // not in map = supported
    }

    #[test]
    fn test_clamp_prefers_downgrade() {
        let map = HashMap::from([
            ("off".into(), None),
            ("low".into(), None),
            ("medium".into(), None),
            ("high".into(), Some("high".into())),
        ]);
        let model = reasoning_model(Some(map));
        // Request medium (disabled) → should downgrade to minimal (available)
        let result = clamp_thinking_level(&model, &ModelThinkingLevel::Medium);
        assert_eq!(result, ModelThinkingLevel::Minimal);
    }

    #[test]
    fn test_clamp_upgrades_when_no_lower() {
        let map = HashMap::from([
            ("off".into(), None),
            ("minimal".into(), None),
            ("low".into(), None),
            ("medium".into(), None),
            ("high".into(), Some("high".into())),
        ]);
        let model = reasoning_model(Some(map));
        // Only high is available → must upgrade
        let result = clamp_thinking_level(&model, &ModelThinkingLevel::Low);
        assert_eq!(result, ModelThinkingLevel::High);
    }

    #[test]
    fn test_map_thinking_level() {
        let map = HashMap::from([
            ("high".into(), Some("custom_value".into())),
        ]);
        let model = reasoning_model(Some(map));
        let mapped = map_thinking_level(&model, &ModelThinkingLevel::High);
        assert_eq!(mapped, Some("custom_value".into()));
    }

    #[test]
    fn test_supports_xhigh() {
        let model_no = reasoning_model(None);
        assert!(!supports_xhigh(&model_no));

        let map = HashMap::from([("xhigh".into(), Some("max".into()))]);
        let model_yes = reasoning_model(Some(map));
        assert!(supports_xhigh(&model_yes));
    }

    #[test]
    fn test_clamp_reasoning() {
        assert_eq!(clamp_reasoning(&ThinkingLevel::XHigh), ThinkingLevel::High);
        assert_eq!(clamp_reasoning(&ThinkingLevel::Medium), ThinkingLevel::Medium);
    }

    #[test]
    fn test_adjust_max_tokens() {
        let budgets = default_thinking_budgets();
        let (max, budget) = adjust_max_tokens_for_thinking(4096, 16384, &ThinkingLevel::Medium, &budgets);
        assert!(max <= 16384);
        assert!(budget > 0);
        assert!(budget <= max);
    }

    #[test]
    fn test_calculate_cost() {
        let model = reasoning_model(None);
        let model = Model { cost: ModelCost { input: 3.0, output: 15.0, ..Default::default() }, ..model };
        let usage = crate::types::Usage { input: 1000, output: 500, ..Default::default() };
        let cost = calculate_cost(&model, &usage);
        assert!((cost.input - 0.003).abs() < 0.0001);
        assert!((cost.output - 0.0075).abs() < 0.0001);
    }
}
