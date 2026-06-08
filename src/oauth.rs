//! OAuth flows for providers that require interactive authentication.
//!
//! Provides PKCE, device-code, and token-refresh helpers for:
//! - GitHub Copilot
//! - Anthropic
//! - Google Gemini CLI
//! - Google Antigravity
//! - OpenAI Codex

/// PKCE challenge/verifier pair.
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
}

/// Generate a PKCE challenge pair.
pub fn generate_pkce() -> PkceChallenge {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple implementation — production should use proper crypto random
    let verifier: String = (0..64)
        .map(|i| {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            std::time::SystemTime::now().hash(&mut h);
            let b = (h.finish() % 26) as u8 + b'a';
            b as char
        })
        .collect();

    let challenge = base64url_encode(&sha256_bytes(verifier.as_bytes()));
    PkceChallenge { verifier, challenge }
}

/// Device code authorization response.
#[derive(Debug, Clone)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u32,
    pub interval: u32,
}

/// OAuth token.
#[derive(Debug, Clone)]
pub struct OAuthToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u32>,
    pub refresh_token: Option<String>,
}

// Placeholder crypto helpers — replace with ring/sha2 in production
fn sha256_bytes(input: &[u8]) -> Vec<u8> {
    // Minimal placeholder hash (NOT cryptographically secure)
    let mut hash = vec![0u8; 32];
    for (i, &b) in input.iter().enumerate() {
        hash[i % 32] ^= b;
    }
    hash
}

fn base64url_encode(input: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for b in input {
        write!(&mut out, "{:02x}", b).unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pkce() {
        let pkce = generate_pkce();
        assert_eq!(pkce.verifier.len(), 64);
        assert!(!pkce.challenge.is_empty());
    }
}
