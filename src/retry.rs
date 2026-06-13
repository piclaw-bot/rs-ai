//! HTTP retry logic with exponential backoff.

use std::time::Duration;

/// Retry configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter_fraction: f64,
    pub max_retry_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter_fraction: 0.25,
            max_retry_delay_ms: 60_000,
        }
    }
}

impl RetryConfig {
    /// No retries.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }
}

/// Compute exponential backoff delay for an attempt.
pub fn compute_backoff(attempt: u32, config: &RetryConfig) -> Duration {
    let base = config.initial_delay.as_secs_f64()
        * config.backoff_multiplier.powi(attempt as i32);
    let capped = base.min(config.max_delay.as_secs_f64());
    // Simple jitter: multiply by (1 - jitter/2) for deterministic tests
    let jittered = capped * (1.0 - config.jitter_fraction * 0.5);
    Duration::from_secs_f64(jittered.max(0.0))
}

/// Check if an HTTP status code is retryable.
pub fn is_retryable_status(code: u16) -> bool {
    matches!(code, 429 | 500 | 502 | 503 | 504)
}

/// Parse a `Retry-After` header value (integer/float seconds, or HTTP-date) into a Duration.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if let Ok(seconds) = trimmed.parse::<f64>() {
        if seconds.is_finite() && seconds >= 0.0 {
            return Some(Duration::from_secs_f64(seconds));
        }
        return None;
    }
    // HTTP-date form: delay until that instant.
    if let Ok(when) = httpdate::parse_http_date(trimmed) {
        return Some(when.duration_since(std::time::SystemTime::now()).unwrap_or(Duration::ZERO));
    }
    None
}

/// Resolve a retry delay from response headers, preferring `retry-after-ms`
/// then `retry-after` (mirrors upstream getRetryAfterDelayMs).
pub fn retry_after_delay(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok())
        && let Ok(millis) = ms.trim().parse::<f64>()
        && millis.is_finite() {
        return Some(Duration::from_millis(millis.max(0.0) as u64));
    }
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use wiremock::matchers::{method, path};

    #[test]
    fn test_compute_backoff() {
        let config = RetryConfig::default();
        let d0 = compute_backoff(0, &config);
        let d1 = compute_backoff(1, &config);
        assert!(d1 > d0, "backoff should increase");
        assert!(d1.as_secs_f64() <= config.max_delay.as_secs_f64());
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable_status(429));
        assert!(is_retryable_status(500));
        assert!(!is_retryable_status(200));
        assert!(!is_retryable_status(400));
    }

    #[test]
    fn test_parse_retry_after() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("1.5"), Some(Duration::from_secs_f64(1.5)));
        assert_eq!(parse_retry_after("not-a-number"), None);
        // HTTP-date in the past clamps to zero.
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), Some(Duration::ZERO));
    }

    #[test]
    fn test_retry_after_delay_prefers_ms_header() {
        use reqwest::header::HeaderMap;
        let mut h = HeaderMap::new();
        h.insert("retry-after-ms", "250".parse().unwrap());
        h.insert(reqwest::header::RETRY_AFTER, "5".parse().unwrap());
        // retry-after-ms wins over retry-after.
        assert_eq!(retry_after_delay(&h), Some(Duration::from_millis(250)));
        let mut h2 = HeaderMap::new();
        h2.insert(reqwest::header::RETRY_AFTER, "5".parse().unwrap());
        assert_eq!(retry_after_delay(&h2), Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_retry_config_from_options() {
        let opts = crate::types::StreamOptions {
            max_retries: Some(4),
            max_retry_delay_ms: Some(1500),
            ..Default::default()
        };
        let cfg = retry_config_from_options(&opts);
        assert_eq!(cfg.max_retries, 4);
        assert_eq!(cfg.max_retry_delay_ms, 1500);
        assert_eq!(cfg.max_delay, Duration::from_millis(1500));
    }

    #[tokio::test]
    async fn test_do_with_retry_retries_retryable_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/retry"))
            .respond_with(ResponseTemplate::new(503).set_body_string("busy"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/retry"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let req = client.get(format!("{}/retry", server.uri()));
        let cfg = RetryConfig {
            max_retries: 1,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            backoff_multiplier: 1.0,
            jitter_fraction: 0.0,
            max_retry_delay_ms: 5,
        };
        let resp = do_with_retry(&client, req, &cfg).await.unwrap();
        assert_eq!(resp.status(), 200);
    }
}

/// No-retry config.
pub fn no_retry_config() -> RetryConfig {
    RetryConfig::none()
}

/// Default retry config.
pub fn default_retry_config() -> RetryConfig {
    RetryConfig::default()
}

/// Build retry config from stream options (mirrors Go's RetryConfigFromOptions).
pub fn retry_config_from_options(opts: &crate::types::StreamOptions) -> RetryConfig {
    if opts.retry_config.is_none() && opts.max_retries.is_none() && opts.max_retry_delay_ms.is_none() {
        return RetryConfig::none();
    }

    let mut cfg = opts.retry_config.clone().unwrap_or_default();
    if let Some(max_retries) = opts.max_retries {
        cfg.max_retries = max_retries;
    }
    if let Some(max_retry_delay_ms) = opts.max_retry_delay_ms {
        cfg.max_retry_delay_ms = max_retry_delay_ms;
        cfg.max_delay = Duration::from_millis(max_retry_delay_ms);
    }
    cfg
}

/// Execute an HTTP request with retry logic (async).
pub async fn do_with_retry(
    _client: &reqwest::Client,
    request_builder: reqwest::RequestBuilder,
    config: &RetryConfig,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut attempt = 0u32;
    let mut builder = request_builder;

    loop {
        let retry_builder = builder.try_clone();
        match builder.send().await {
            Ok(resp) => {
                if !is_retryable_status(resp.status().as_u16()) || attempt >= config.max_retries {
                    return Ok(resp);
                }

                let retry_after = retry_after_delay(resp.headers());
                let mut delay = retry_after.unwrap_or_else(|| compute_backoff(attempt, config));
                delay = delay.min(Duration::from_millis(config.max_retry_delay_ms));
                tokio::time::sleep(delay).await;

                attempt += 1;
                builder = match retry_builder {
                    Some(b) => b,
                    None => return Ok(resp),
                };
            }
            Err(err) => {
                if attempt >= config.max_retries {
                    return Err(err);
                }
                let delay = compute_backoff(attempt, config).min(Duration::from_millis(config.max_retry_delay_ms));
                tokio::time::sleep(delay).await;
                attempt += 1;
                builder = match retry_builder {
                    Some(b) => b,
                    None => return Err(err),
                };
            }
        }
    }
}
