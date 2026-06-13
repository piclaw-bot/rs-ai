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
            compat: Default::default(),
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
    fn test_clamp_prefers_upgrade() {
        let map = HashMap::from([
            ("off".into(), None),
            ("low".into(), None),
            ("medium".into(), None),
            ("high".into(), Some("high".into())),
        ]);
        let model = reasoning_model(Some(map));
        // Request medium (disabled). Available = [minimal, high].
        // Upstream clamps upward first -> high.
        let result = clamp_thinking_level(&model, &ModelThinkingLevel::Medium);
        assert_eq!(result, ModelThinkingLevel::High);
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

    #[test]
    fn test_map_openai_finish_reason() {
        use crate::types::StopReason;
        assert_eq!(map_openai_finish_reason("stop").0, StopReason::Stop);
        assert_eq!(map_openai_finish_reason("end").0, StopReason::Stop);
        assert_eq!(map_openai_finish_reason("length").0, StopReason::Length);
        assert_eq!(map_openai_finish_reason("function_call").0, StopReason::ToolUse);
        assert_eq!(map_openai_finish_reason("tool_calls").0, StopReason::ToolUse);
        let (r, msg) = map_openai_finish_reason("content_filter");
        assert_eq!(r, StopReason::Error);
        assert!(msg.unwrap().contains("content_filter"));
        let (r2, msg2) = map_openai_finish_reason("some_unknown");
        assert_eq!(r2, StopReason::Error);
        assert!(msg2.unwrap().contains("some_unknown"));
    }

    #[test]
    fn test_parse_openai_usage_subtracts_cache_and_computes_cost() {
        let model = reasoning_model(None);
        let model = Model { cost: ModelCost { input: 3.0, output: 15.0, cache_read: 0.3, cache_write: 0.0 }, ..model };
        let raw = serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": { "cached_tokens": 400 }
        });
        let usage = parse_openai_usage(&raw, &model);
        assert_eq!(usage.cache_read, 400);
        assert_eq!(usage.input, 600); // 1000 - 400 cached
        assert_eq!(usage.output, 200);
        assert_eq!(usage.total_tokens, 1200); // 600 + 200 + 400
        // cost: input 600*3/1e6 + output 200*15/1e6 + cache_read 400*0.3/1e6
        assert!((usage.cost.input - 0.0018).abs() < 1e-6);
        assert!((usage.cost.cache_read - 0.00012).abs() < 1e-7);
    }

    #[test]
    fn test_parse_responses_usage_cache() {
        let model = reasoning_model(None);
        let raw = serde_json::json!({
            "input_tokens": 500, "output_tokens": 100, "total_tokens": 600,
            "input_tokens_details": { "cached_tokens": 200 }
        });
        let usage = parse_responses_usage(&raw, &model);
        assert_eq!(usage.cache_read, 200);
        assert_eq!(usage.input, 300); // 500 - 200
        assert_eq!(usage.output, 100);
    }

    #[test]
    fn test_apply_service_tier_pricing() {
        let mut model = reasoning_model(None);
        model.cost = ModelCost { input: 1.0, output: 2.0, cache_read: 0.5, cache_write: 0.0 };
        let base = crate::types::Usage {
            input: 1_000_000, output: 1_000_000, cache_read: 1_000_000, cache_write: 0,
            total_tokens: 3_000_000, cost: Default::default(),
        };
        // flex halves the cost.
        let mut u = base.clone();
        u.cost = calculate_cost(&model, &u);
        let full_total = u.cost.total;
        apply_service_tier_pricing(&model, &mut u, Some("flex"));
        assert!((u.cost.total - full_total * 0.5).abs() < 1e-9);
        // priority doubles.
        let mut u = base.clone();
        u.cost = calculate_cost(&model, &u);
        apply_service_tier_pricing(&model, &mut u, Some("priority"));
        assert!((u.cost.total - full_total * 2.0).abs() < 1e-9);
        // default tier leaves cost unchanged.
        let mut u = base.clone();
        u.cost = calculate_cost(&model, &u);
        apply_service_tier_pricing(&model, &mut u, None);
        assert!((u.cost.total - full_total).abs() < 1e-9);
    }
}
