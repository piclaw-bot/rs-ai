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

/// Anthropic OAuth client id (decoded from the upstream base64 constant).
pub const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// Anthropic OAuth token endpoint.
pub const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";

/// A refreshed OAuth token.
#[derive(Debug, Clone)]
pub struct RefreshedToken {
    pub access: String,
    pub refresh: Option<String>,
    /// Absolute expiry in epoch milliseconds, with a 5-minute safety margin.
    pub expires_at_ms: i64,
}

/// Refresh an Anthropic OAuth token (mirrors refreshAnthropicToken).
pub async fn refresh_anthropic_token(refresh_token: &str) -> Result<RefreshedToken, String> {
    refresh_anthropic_token_at(ANTHROPIC_TOKEN_URL, refresh_token).await
}

/// Refresh against an explicit token endpoint (used for testing).
pub async fn refresh_anthropic_token_at(token_url: &str, refresh_token: &str) -> Result<RefreshedToken, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(token_url)
        .json(&serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_CLIENT_ID,
            "refresh_token": refresh_token,
        }))
        .send()
        .await
        .map_err(|e| format!("Anthropic token refresh request failed. url={token_url}; details={e}"))?;
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Anthropic token refresh request failed. url={token_url}; details={e}"))?;
    let data: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("Anthropic token refresh returned invalid JSON. url={token_url}; body={body}; details={e}"))?;
    let access = data
        .get("access_token")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Anthropic token refresh missing access_token. body={body}"))?
        .to_string();
    let refresh = data.get("refresh_token").and_then(|v| v.as_str()).map(|s| s.to_string());
    let expires_in = data.get("expires_in").and_then(|v| v.as_i64()).unwrap_or(0);
    let expires_at_ms = crate::utils::now_millis() + expires_in * 1000 - 5 * 60 * 1000;
    Ok(RefreshedToken { access, refresh, expires_at_ms })
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

    #[tokio::test]
    async fn test_refresh_anthropic_token() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::{method, path, body_partial_json};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .and(body_partial_json(serde_json::json!({"grant_type": "refresh_token", "refresh_token": "old-refresh"})))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#,
            ))
            .mount(&server)
            .await;
        let url = format!("{}/oauth/token", server.uri());
        let before = crate::utils::now_millis();
        let tok = refresh_anthropic_token_at(&url, "old-refresh").await.unwrap();
        assert_eq!(tok.access, "new-access");
        assert_eq!(tok.refresh.as_deref(), Some("new-refresh"));
        // expires ~= now + 3600s - 5min safety margin.
        let expected = before + 3600 * 1000 - 5 * 60 * 1000;
        assert!((tok.expires_at_ms - expected).abs() < 5000);
    }

    #[tokio::test]
    async fn test_refresh_anthropic_token_invalid_json() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>nope</html>"))
            .mount(&server)
            .await;
        let err = refresh_anthropic_token_at(&server.uri(), "r").await.unwrap_err();
        assert!(err.contains("invalid JSON"));
    }
}
