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

/// Parse Retry-After header value into a Duration.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if let Ok(seconds) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(parse_retry_after("not-a-number"), None);
    }
}
