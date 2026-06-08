//! Pluggable leveled logging.

use std::sync::OnceLock;

/// Log level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Logger trait.
pub trait Logger: Send + Sync {
    fn log(&self, level: LogLevel, msg: &str, fields: &[(&str, &str)]);
}

/// A simple stderr logger.
pub struct StderrLogger {
    min_level: LogLevel,
}

impl StderrLogger {
    pub fn new(min_level: LogLevel) -> Self {
        Self { min_level }
    }
}

impl Logger for StderrLogger {
    fn log(&self, level: LogLevel, msg: &str, fields: &[(&str, &str)]) {
        if level < self.min_level {
            return;
        }
        let level_str = match level {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        };
        let fields_str: String = fields
            .iter()
            .map(|(k, v)| format!(" {}={}", k, v))
            .collect();
        eprintln!("[rs-ai] {} {}{}", level_str, msg, fields_str);
    }
}

static LOGGER: OnceLock<Box<dyn Logger>> = OnceLock::new();

/// Set the global logger.
pub fn set_logger(logger: Box<dyn Logger>) {
    let _ = LOGGER.set(logger);
}

/// Get the global logger (or a no-op).
pub fn get_logger() -> &'static dyn Logger {
    LOGGER.get().map(|l| l.as_ref()).unwrap_or(&NoopLogger)
}

struct NoopLogger;
impl Logger for NoopLogger {
    fn log(&self, _: LogLevel, _: &str, _: &[(&str, &str)]) {}
}

/// Log helpers.
pub fn log_debug(msg: &str, fields: &[(&str, &str)]) { get_logger().log(LogLevel::Debug, msg, fields); }
pub fn log_info(msg: &str, fields: &[(&str, &str)]) { get_logger().log(LogLevel::Info, msg, fields); }
pub fn log_warn(msg: &str, fields: &[(&str, &str)]) { get_logger().log(LogLevel::Warn, msg, fields); }
pub fn log_error(msg: &str, fields: &[(&str, &str)]) { get_logger().log(LogLevel::Error, msg, fields); }

/// Create a new stderr logger (alias for convenience).
pub fn new_stderr_logger(min_level: LogLevel) -> StderrLogger {
    StderrLogger::new(min_level)
}

/// Simple logger alias (same as StderrLogger).
pub fn new_simple_logger(min_level: LogLevel) -> StderrLogger {
    StderrLogger::new(min_level)
}
