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

/// GitHub Copilot OAuth client id (decoded from the upstream base64 constant).
pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// A GitHub device-code grant returned by `start_github_device_flow`.
#[derive(Debug, Clone)]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval: Option<u64>,
    pub expires_in: u64,
}

/// Start a GitHub device-code flow (mirrors startDeviceFlow).
pub async fn start_github_device_flow(domain: &str) -> Result<DeviceCode, String> {
    start_github_device_flow_at(&format!("https://{domain}/login/device/code"), COPILOT_CLIENT_ID).await
}

/// Start against an explicit device-code endpoint (used for testing).
pub async fn start_github_device_flow_at(device_code_url: &str, client_id: &str) -> Result<DeviceCode, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(device_code_url)
        .header("Accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .form(&[("client_id", client_id), ("scope", "read:user")])
        .send()
        .await
        .map_err(|e| format!("Device code request failed: {e}"))?;
    let body = resp.text().await.map_err(|e| format!("Device code request failed: {e}"))?;
    let data: serde_json::Value = serde_json::from_str(&body)
        .map_err(|_| "Invalid device code response".to_string())?;
    let device_code = data.get("device_code").and_then(|v| v.as_str())
        .ok_or_else(|| "Invalid device code response fields".to_string())?.to_string();
    let user_code = data.get("user_code").and_then(|v| v.as_str())
        .ok_or_else(|| "Invalid device code response fields".to_string())?.to_string();
    let verification_uri = data.get("verification_uri").and_then(|v| v.as_str())
        .ok_or_else(|| "Invalid device code response fields".to_string())?.to_string();
    let expires_in = data.get("expires_in").and_then(|v| v.as_u64())
        .ok_or_else(|| "Invalid device code response fields".to_string())?;
    // Reject non-http(s) verification URIs to avoid opening arbitrary handlers.
    if !(verification_uri.starts_with("https://") || verification_uri.starts_with("http://")) {
        return Err("Untrusted verification_uri in device code response".to_string());
    }
    let interval = data.get("interval").and_then(|v| v.as_u64());
    Ok(DeviceCode { device_code, user_code, verification_uri, interval, expires_in })
}

/// Result of a single device-code token poll (mirrors the poll callback classification).
#[derive(Debug, Clone, PartialEq)]
pub enum DevicePollStatus {
    Complete(String),
    Pending,
    SlowDown,
    Failed(String),
}

/// Poll once for a GitHub device-code access token (mirrors pollForGitHubAccessToken's poll).
pub async fn poll_github_device_token(domain: &str, device_code: &str) -> DevicePollStatus {
    poll_github_device_token_at(&format!("https://{domain}/login/oauth/access_token"), COPILOT_CLIENT_ID, device_code).await
}

/// Poll against an explicit access-token endpoint (used for testing).
pub async fn poll_github_device_token_at(access_token_url: &str, client_id: &str, device_code: &str) -> DevicePollStatus {
    let client = reqwest::Client::new();
    let resp = client
        .post(access_token_url)
        .header("Accept", "application/json")
        .header("User-Agent", "GitHubCopilotChat/0.35.0")
        .form(&[
            ("client_id", client_id),
            ("device_code", device_code),
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
        ])
        .send()
        .await;
    let body = match resp {
        Ok(r) => r.text().await.unwrap_or_default(),
        Err(e) => return DevicePollStatus::Failed(format!("Device flow failed: {e}")),
    };
    let data: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => return DevicePollStatus::Failed("Invalid device token response".to_string()),
    };
    if let Some(token) = data.get("access_token").and_then(|v| v.as_str()) {
        return DevicePollStatus::Complete(token.to_string());
    }
    if let Some(error) = data.get("error").and_then(|v| v.as_str()) {
        return match error {
            "authorization_pending" => DevicePollStatus::Pending,
            "slow_down" => DevicePollStatus::SlowDown,
            other => {
                let desc = data.get("error_description").and_then(|v| v.as_str())
                    .map(|d| format!(": {d}")).unwrap_or_default();
                DevicePollStatus::Failed(format!("Device flow failed: {other}{desc}"))
            }
        };
    }
    DevicePollStatus::Failed("Invalid device token response".to_string())
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

    #[tokio::test]
    async fn test_start_github_device_flow() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::{method, path};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login/device/code"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"device_code":"dc","user_code":"WXYZ-1234","verification_uri":"https://github.com/login/device","interval":5,"expires_in":900}"#,
            ))
            .mount(&server)
            .await;
        let url = format!("{}/login/device/code", server.uri());
        let dc = start_github_device_flow_at(&url, COPILOT_CLIENT_ID).await.unwrap();
        assert_eq!(dc.device_code, "dc");
        assert_eq!(dc.user_code, "WXYZ-1234");
        assert_eq!(dc.verification_uri, "https://github.com/login/device");
        assert_eq!(dc.interval, Some(5));
        assert_eq!(dc.expires_in, 900);
    }

    #[tokio::test]
    async fn test_start_github_device_flow_rejects_untrusted_uri() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::method;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                r#"{"device_code":"dc","user_code":"x","verification_uri":"javascript:alert(1)","expires_in":900}"#,
            ))
            .mount(&server)
            .await;
        let err = start_github_device_flow_at(&server.uri(), COPILOT_CLIENT_ID).await.unwrap_err();
        assert!(err.contains("Untrusted verification_uri"));
    }

    #[tokio::test]
    async fn test_poll_github_device_token_classifies_responses() {
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use wiremock::matchers::{method, path};
        // pending
        let s1 = MockServer::start().await;
        Mock::given(method("POST")).and(path("/t"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"error":"authorization_pending"}"#))
            .mount(&s1).await;
        assert_eq!(poll_github_device_token_at(&format!("{}/t", s1.uri()), COPILOT_CLIENT_ID, "dc").await, DevicePollStatus::Pending);
        // slow_down
        let s2 = MockServer::start().await;
        Mock::given(method("POST")).and(path("/t"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"error":"slow_down"}"#))
            .mount(&s2).await;
        assert_eq!(poll_github_device_token_at(&format!("{}/t", s2.uri()), COPILOT_CLIENT_ID, "dc").await, DevicePollStatus::SlowDown);
        // complete
        let s3 = MockServer::start().await;
        Mock::given(method("POST")).and(path("/t"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"access_token":"gho_tok"}"#))
            .mount(&s3).await;
        assert_eq!(poll_github_device_token_at(&format!("{}/t", s3.uri()), COPILOT_CLIENT_ID, "dc").await, DevicePollStatus::Complete("gho_tok".to_string()));
        // failed with description
        let s4 = MockServer::start().await;
        Mock::given(method("POST")).and(path("/t"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"error":"access_denied","error_description":"nope"}"#))
            .mount(&s4).await;
        assert_eq!(poll_github_device_token_at(&format!("{}/t", s4.uri()), COPILOT_CLIENT_ID, "dc").await, DevicePollStatus::Failed("Device flow failed: access_denied: nope".to_string()));
    }
}
