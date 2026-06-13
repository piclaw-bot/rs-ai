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

/// OpenAI Codex OAuth client id.
pub const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
/// OpenAI Codex OAuth token endpoint.
pub const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// Refreshed Codex credentials (includes the ChatGPT account id from the JWT).
#[derive(Debug, Clone)]
pub struct CodexCredentials {
    pub access: String,
    pub refresh: Option<String>,
    pub expires_at_ms: i64,
    pub account_id: String,
}

/// Decode a JWT payload (middle segment) into JSON.
fn decode_jwt_payload(token: &str) -> Option<serde_json::Value> {
    use base64::Engine;
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts[1])
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(parts[1]))
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Refresh an OpenAI Codex OAuth token (mirrors refreshOpenAICodexToken).
pub async fn refresh_codex_token(refresh_token: &str) -> Result<CodexCredentials, String> {
    refresh_codex_token_at(CODEX_TOKEN_URL, refresh_token).await
}

/// Refresh against an explicit token endpoint (used for testing).
pub async fn refresh_codex_token_at(token_url: &str, refresh_token: &str) -> Result<CodexCredentials, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(token_url)
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", CODEX_CLIENT_ID),
        ])
        .send()
        .await
        .map_err(|e| format!("OpenAI Codex token refresh error: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.map_err(|e| format!("OpenAI Codex token refresh error: {e}"))?;
    if !status.is_success() {
        return Err(format!("OpenAI Codex token refresh failed ({status}): {body}"));
    }
    let data: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("OpenAI Codex token refresh invalid JSON: body={body}; details={e}"))?;
    let access = data.get("access_token").and_then(|v| v.as_str())
        .ok_or_else(|| format!("OpenAI Codex token refresh response missing fields: {body}"))?
        .to_string();
    let refresh = data.get("refresh_token").and_then(|v| v.as_str()).map(|s| s.to_string());
    let expires_in = data.get("expires_in").and_then(|v| v.as_i64())
        .ok_or_else(|| format!("OpenAI Codex token refresh response missing fields: {body}"))?;
    let account_id = decode_jwt_payload(&access)
        .and_then(|p| p.get(CODEX_JWT_CLAIM_PATH).and_then(|a| a.get("chatgpt_account_id")).and_then(|v| v.as_str()).map(|s| s.to_string()))
        .ok_or_else(|| "Failed to extract accountId from token".to_string())?;
    Ok(CodexCredentials {
        access,
        refresh,
        expires_at_ms: crate::utils::now_millis() + expires_in * 1000,
        account_id,
    })
}

/// Refreshed GitHub Copilot credentials.
#[derive(Debug, Clone)]
pub struct CopilotCredentials {
    pub access: String,
    pub refresh: String,
    pub expires_at_ms: i64,
}

/// The Copilot token-exchange URL for a domain (default github.com).
pub fn copilot_token_url(domain: &str) -> String {
    format!("https://api.{domain}/copilot_internal/v2/token")
}

/// Refresh a GitHub Copilot token (mirrors refreshGitHubCopilotToken).
pub async fn refresh_copilot_token(refresh_token: &str, enterprise_domain: Option<&str>) -> Result<CopilotCredentials, String> {
    let domain = enterprise_domain.unwrap_or("github.com");
    refresh_copilot_token_at(&copilot_token_url(domain), refresh_token).await
}

/// Refresh against an explicit token endpoint (used for testing).
pub async fn refresh_copilot_token_at(token_url: &str, refresh_token: &str) -> Result<CopilotCredentials, String> {
    let client = reqwest::Client::new();
    let mut req = client
        .get(token_url)
        .header("Accept", "application/json")
        .header("Authorization", format!("Bearer {refresh_token}"));
    for (k, v) in crate::utils::copilot_headers() {
        req = req.header(k, v);
    }
    let resp = req.send().await.map_err(|e| format!("Copilot token refresh error: {e}"))?;
    let body = resp.text().await.map_err(|e| format!("Copilot token refresh error: {e}"))?;
    let data: serde_json::Value = serde_json::from_str(&body)
        .map_err(|_| "Invalid Copilot token response".to_string())?;
    let token = data.get("token").and_then(|v| v.as_str())
        .ok_or_else(|| "Invalid Copilot token response fields".to_string())?;
    let expires_at = data.get("expires_at").and_then(|v| v.as_i64())
        .ok_or_else(|| "Invalid Copilot token response fields".to_string())?;
    Ok(CopilotCredentials {
        access: token.to_string(),
        refresh: refresh_token.to_string(),
        expires_at_ms: expires_at * 1000 - 5 * 60 * 1000,
    })
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

    #[tokio::test]
    async fn test_refresh_codex_token_extracts_account_id() {
        use base64::Engine;
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::{method, path};
        // Build a JWT whose payload carries the chatgpt_account_id claim.
        let payload = serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc_123"}
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let jwt = format!("h.{payload_b64}.s");
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!(
                r#"{{"access_token":"{jwt}","refresh_token":"new-refresh","expires_in":3600}}"#
            )))
            .mount(&server)
            .await;
        let url = format!("{}/oauth/token", server.uri());
        let creds = refresh_codex_token_at(&url, "old").await.unwrap();
        assert_eq!(creds.access, jwt);
        assert_eq!(creds.refresh.as_deref(), Some("new-refresh"));
        assert_eq!(creds.account_id, "acc_123");
    }

    #[tokio::test]
    async fn test_refresh_copilot_token() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::{method, path, header};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/copilot_internal/v2/token"))
            .and(header("Authorization", "Bearer gho_refresh"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"token":"copilot-access","expires_at":1000000}"#,
            ))
            .mount(&server)
            .await;
        let url = format!("{}/copilot_internal/v2/token", server.uri());
        let creds = refresh_copilot_token_at(&url, "gho_refresh").await.unwrap();
        assert_eq!(creds.access, "copilot-access");
        assert_eq!(creds.refresh, "gho_refresh");
        // expires_at (seconds) * 1000 - 5min margin.
        assert_eq!(creds.expires_at_ms, 1000000 * 1000 - 5 * 60 * 1000);
    }
}
