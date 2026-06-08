//! OpenAI-compatible provider compatibility detection.

use crate::types::Model;

/// OpenAI Completions compatibility overrides.
#[derive(Debug, Clone, Default)]
pub struct OpenAICompletionsCompat {
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
    pub supports_usage_in_streaming: Option<bool>,
    pub supports_temperature: Option<bool>,
    pub max_tokens_field: Option<String>,
    pub requires_tool_result_name: Option<bool>,
    pub requires_thinking_as_text: Option<bool>,
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    pub thinking_format: Option<String>,
    pub supports_strict_mode: Option<bool>,
    pub supports_long_cache_retention: Option<bool>,
}

/// Auto-detect compatibility flags from a model's provider/URL.
pub fn detect_compat(model: &Model) -> OpenAICompletionsCompat {
    detect_compat_inner(&model.provider, &model.id, &model.base_url)
}

/// Detect and merge model-specific compat overrides (mirrors Go's DetectCompatForModel).
pub fn detect_compat_for_model(model: &Model, overrides: Option<&OpenAICompletionsCompat>) -> OpenAICompletionsCompat {
    let mut c = detect_compat_inner(&model.provider, &model.id, &model.base_url);
    if let Some(o) = overrides {
        if o.supports_store.is_some() { c.supports_store = o.supports_store; }
        if o.supports_developer_role.is_some() { c.supports_developer_role = o.supports_developer_role; }
        if o.supports_reasoning_effort.is_some() { c.supports_reasoning_effort = o.supports_reasoning_effort; }
        if o.supports_usage_in_streaming.is_some() { c.supports_usage_in_streaming = o.supports_usage_in_streaming; }
        if o.supports_temperature.is_some() { c.supports_temperature = o.supports_temperature; }
        if o.max_tokens_field.is_some() { c.max_tokens_field = o.max_tokens_field.clone(); }
        if o.requires_tool_result_name.is_some() { c.requires_tool_result_name = o.requires_tool_result_name; }
        if o.requires_thinking_as_text.is_some() { c.requires_thinking_as_text = o.requires_thinking_as_text; }
        if o.requires_reasoning_content_on_assistant_messages.is_some() { c.requires_reasoning_content_on_assistant_messages = o.requires_reasoning_content_on_assistant_messages; }
        if o.thinking_format.is_some() { c.thinking_format = o.thinking_format.clone(); }
        if o.supports_strict_mode.is_some() { c.supports_strict_mode = o.supports_strict_mode; }
        if o.supports_long_cache_retention.is_some() { c.supports_long_cache_retention = o.supports_long_cache_retention; }
    }
    c
}

fn detect_compat_inner(provider: &str, model_id: &str, base_url: &str) -> OpenAICompletionsCompat {
    let mut c = OpenAICompletionsCompat {
        supports_store: Some(true),
        supports_developer_role: Some(true),
        supports_reasoning_effort: Some(true),
        supports_usage_in_streaming: Some(true),
        supports_temperature: Some(true),
        max_tokens_field: Some("max_completion_tokens".to_string()),
        supports_strict_mode: Some(true),
        supports_long_cache_retention: Some(true),
        ..Default::default()
    };

    let is_ollama = is_local_ollama(base_url);
    let is_openrouter = provider == "openrouter" || base_url.contains("openrouter.ai");
    let is_deepseek = provider == "deepseek" || base_url.contains("deepseek.com")
        || provider == "xiaomi" || base_url.contains("xiaomimimo.com");
    let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
    let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
    let is_zai = provider == "zai" || provider == "zai-coding-cn"
        || base_url.contains("z.ai") || base_url.contains("open.bigmodel.cn");
    let is_moonshot = provider == "moonshotai" || provider == "moonshotai-cn"
        || base_url.contains("api.moonshot.");
    let is_cloudflare_aigw = provider == "cloudflare-ai-gateway"
        || base_url.contains("gateway.ai.cloudflare.com");

    let is_non_standard = provider == "cerebras" || provider == "xai"
        || base_url.contains("chutes.ai") || is_deepseek || is_zai || is_moonshot
        || provider == "opencode" || base_url.contains("opencode.ai")
        || provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com")
        || is_cloudflare_aigw || is_ollama || is_nvidia || is_ant_ling;

    let use_max_tokens = base_url.contains("chutes.ai") || is_moonshot
        || is_cloudflare_aigw || is_ollama || is_nvidia || is_ant_ling;

    if is_non_standard {
        c.supports_store = Some(false);
        c.supports_developer_role = Some(false);
    }
    if use_max_tokens {
        c.max_tokens_field = Some("max_tokens".to_string());
    }
    if is_openrouter {
        c.thinking_format = Some("openrouter".to_string());
        if !model_id.starts_with("anthropic/") && !model_id.starts_with("openai/") {
            c.supports_developer_role = Some(false);
        }
    }
    if is_ollama {
        c.requires_tool_result_name = Some(true);
        c.supports_strict_mode = Some(false);
    }
    if is_zai {
        c.thinking_format = Some("zai".to_string());
    }
    if is_deepseek {
        c.thinking_format = Some("deepseek".to_string());
        c.requires_reasoning_content_on_assistant_messages = Some(true);
    }
    if is_ant_ling {
        c.thinking_format = Some("ant-ling".to_string());
    }
    if is_moonshot || is_cloudflare_aigw {
        c.supports_strict_mode = Some(false);
    }
    if provider == "cloudflare-workers-ai" || is_cloudflare_aigw {
        c.supports_long_cache_retention = Some(false);
    }

    c
}

fn is_local_ollama(url: &str) -> bool {
    if let Ok(parsed) = url::Url::parse(url) {
        let host = parsed.host_str().unwrap_or("");
        let port = parsed.port().unwrap_or(0);
        port == 11434 && (host == "localhost" || host == "127.0.0.1" || host == "[::1]")
    } else {
        false
    }
}
