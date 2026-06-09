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
    use rand::RngCore;

    let mut verifier_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
    let verifier = base64url_encode(&verifier_bytes);
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

fn sha256_bytes(input: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    Sha256::digest(input).to_vec()
}

fn base64url_encode(input: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pkce() {
        let pkce = generate_pkce();
        assert!(!pkce.verifier.is_empty());
        assert!(!pkce.challenge.is_empty());
        assert_ne!(pkce.verifier, pkce.challenge);
        assert!(pkce.verifier.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        assert!(pkce.challenge.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }
}
