//! Environment-based API key resolution.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::types::{Model, StreamOptions};

static ENV_MAP: LazyLock<HashMap<&'static str, &'static [&'static str]>> = LazyLock::new(|| {
    HashMap::from([
        ("github-copilot", &["COPILOT_GITHUB_TOKEN"][..]),
        // ANTHROPIC_OAUTH_TOKEN takes precedence over ANTHROPIC_API_KEY
        ("anthropic", &["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"][..]),
        ("ant-ling", &["ANT_LING_API_KEY"][..]),
        ("openai", &["OPENAI_API_KEY"][..]),
        ("azure-openai-responses", &["AZURE_OPENAI_API_KEY"][..]),
        ("nvidia", &["NVIDIA_API_KEY"][..]),
        ("deepseek", &["DEEPSEEK_API_KEY"][..]),
        ("google", &["GEMINI_API_KEY"][..]),
        ("google-vertex", &["GOOGLE_CLOUD_API_KEY"][..]),
        ("groq", &["GROQ_API_KEY"][..]),
        ("cerebras", &["CEREBRAS_API_KEY"][..]),
        ("xai", &["XAI_API_KEY"][..]),
        ("openrouter", &["OPENROUTER_API_KEY"][..]),
        ("vercel-ai-gateway", &["AI_GATEWAY_API_KEY"][..]),
        ("zai", &["ZAI_API_KEY"][..]),
        ("zai-coding-cn", &["ZAI_CODING_CN_API_KEY"][..]),
        ("mistral", &["MISTRAL_API_KEY"][..]),
        ("minimax", &["MINIMAX_API_KEY"][..]),
        ("minimax-cn", &["MINIMAX_CN_API_KEY"][..]),
        ("moonshotai", &["MOONSHOT_API_KEY"][..]),
        ("moonshotai-cn", &["MOONSHOT_API_KEY"][..]),
        ("huggingface", &["HF_TOKEN"][..]),
        ("fireworks", &["FIREWORKS_API_KEY"][..]),
        ("together", &["TOGETHER_API_KEY"][..]),
        ("opencode", &["OPENCODE_API_KEY"][..]),
        ("opencode-go", &["OPENCODE_API_KEY"][..]),
        ("kimi-coding", &["KIMI_API_KEY"][..]),
        ("cloudflare-workers-ai", &["CLOUDFLARE_API_KEY"][..]),
        ("cloudflare-ai-gateway", &["CLOUDFLARE_API_KEY"][..]),
        ("xiaomi", &["XIAOMI_API_KEY"][..]),
        ("xiaomi-token-plan-cn", &["XIAOMI_TOKEN_PLAN_CN_API_KEY"][..]),
        ("xiaomi-token-plan-ams", &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"][..]),
        ("xiaomi-token-plan-sgp", &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"][..]),
    ])
});

/// Look up an API key from environment variables for a provider.
pub fn get_env_api_key(provider: &str) -> Option<String> {
    if let Some(vars) = ENV_MAP.get(provider) {
        for var in *vars {
            if let Ok(val) = std::env::var(var)
                && !val.is_empty() {
                    return Some(val);
                }
        }
        return None;
    }
    // Generic fallback: PROVIDER_API_KEY
    let upper: String = provider
        .chars()
        .map(|c| if c == '-' || c == '.' { '_' } else { c.to_ascii_uppercase() })
        .collect();
    std::env::var(format!("{}_API_KEY", upper)).ok().filter(|v| !v.is_empty())
}

/// Resolve API key: explicit option > model-level > environment.
pub fn resolve_api_key(model: &Model, opts: &StreamOptions) -> Option<String> {
    if let Some(ref key) = opts.api_key {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(ref key) = model.api_key {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    get_env_api_key(&model.provider)
}
