//! Diagnostic types for assistant message metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Error captured as a diagnostic without failing the overall request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

/// A diagnostic record attached to an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: String,
    pub timestamp: i64,
    pub error: DiagnosticError,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<HashMap<String, serde_json::Value>>,
}

/// Create a transport failure diagnostic.
pub fn transport_failure_diagnostic(error_msg: &str) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        diagnostic_type: "provider_transport_failure".into(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64,
        error: DiagnosticError {
            name: Some("TransportError".into()),
            message: error_msg.into(),
            stack: None,
            code: None,
        },
        details: None,
    }
}

/// Create an assistant message diagnostic record.
pub fn create_assistant_message_diagnostic(diagnostic_type: &str, error_msg: &str) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        diagnostic_type: diagnostic_type.into(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64,
        error: DiagnosticError {
            name: None,
            message: error_msg.into(),
            stack: None,
            code: None,
        },
        details: None,
    }
}

/// Extract the error message from a diagnostic.
pub fn extract_diagnostic_error(diag: &AssistantMessageDiagnostic) -> &str {
    &diag.error.message
}

/// Append a diagnostic to a message's diagnostic list.
/// (In Rust, Message doesn't have diagnostics field yet — this is a placeholder
/// that returns the diagnostic for the caller to attach.)
pub fn append_assistant_message_diagnostic(diagnostic_type: &str, error_msg: &str) -> AssistantMessageDiagnostic {
    create_assistant_message_diagnostic(diagnostic_type, error_msg)
}
