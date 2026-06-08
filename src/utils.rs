//! Utility helpers: sanitize, hashing, Copilot headers.

use std::collections::HashMap;

/// Sanitize surrogate pairs from a string (replaces unpaired surrogates with replacement char).
pub fn sanitize_surrogates(s: &str) -> String {
    // Rust strings are valid UTF-8 by construction, so surrogates cannot appear.
    // This is a no-op in Rust but exists for API parity with the Go/TS versions.
    s.to_string()
}

/// Simple hash of a string (FNV-1a inspired, for cache keys).
pub fn hash_string(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Generate GitHub Copilot headers.
pub fn copilot_headers() -> HashMap<String, String> {
    HashMap::from([
        ("Copilot-Integration-Id".into(), "vscode-chat".into()),
        ("Editor-Plugin-Version".into(), "copilot-chat/0.35.0".into()),
        ("Editor-Version".into(), "vscode/1.107.0".into()),
        ("User-Agent".into(), "GitHubCopilotChat/0.35.0".into()),
    ])
}

/// Generate GitHub Copilot headers with an intent field.
pub fn copilot_headers_with_intent(intent: &str) -> HashMap<String, String> {
    let mut h = copilot_headers();
    h.insert("openai-intent".into(), intent.into());
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_surrogates_noop() {
        assert_eq!(sanitize_surrogates("Hello 🙈"), "Hello 🙈");
    }

    #[test]
    fn test_hash_string_deterministic() {
        let h1 = hash_string("test");
        let h2 = hash_string("test");
        assert_eq!(h1, h2);
        assert_ne!(h1, hash_string("other"));
    }

    #[test]
    fn test_copilot_headers() {
        let h = copilot_headers();
        assert_eq!(h.get("User-Agent").unwrap(), "GitHubCopilotChat/0.35.0");
    }

    #[test]
    fn test_copilot_headers_with_intent() {
        let h = copilot_headers_with_intent("chat");
        assert_eq!(h.get("openai-intent").unwrap(), "chat");
    }
}

/// Short hash (first 8 hex chars of FNV hash).
pub fn short_hash(s: &str) -> String {
    format!("{:08x}", hash_string(s) & 0xFFFFFFFF)
}

/// Check if a provider is a Cloudflare provider.
pub fn is_cloudflare_provider(provider: &str) -> bool {
    provider == "cloudflare-workers-ai" || provider == "cloudflare-ai-gateway"
}

/// Resolve Cloudflare base URL (pass-through; Cloudflare uses standard paths).
pub fn resolve_cloudflare_base_url(base_url: &str) -> String {
    base_url.to_string()
}

/// Format a thrown/panic value as a string (Rust equivalent: just Display).
pub fn format_thrown_value(err: &dyn std::fmt::Display) -> String {
    err.to_string()
}
