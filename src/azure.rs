//! Azure OpenAI normalization helpers.
//!
//! Handles Azure-specific response format differences (reasoning items,
//! commentary blocks) and session header generation.

use std::collections::HashMap;
use serde_json::Value;

/// Normalize Azure reasoning events to standard format.
///
/// Azure wraps reasoning in `response.output_item.done` with `type: "reasoning"`
/// and a `content` array. This normalizes it to the standard `summary` field.
pub fn normalize_azure_reasoning_event(event: &mut Value) {
    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type != "response.output_item.done" {
        return;
    }
    let item_type = event.pointer("/item/type").and_then(|v| v.as_str()).unwrap_or("");
    if item_type != "reasoning" {
        return;
    }
    // Move content → summary
    if let Some(content) = event.pointer("/item/content").cloned()
        && let Some(item) = event.get_mut("item").and_then(|v| v.as_object_mut()) {
            item.insert("summary".to_string(), content);
            item.remove("content");
        }
}

/// Generate Azure session affinity headers.
pub fn azure_session_headers(session_id: &str) -> HashMap<String, String> {
    if session_id.is_empty() {
        return HashMap::new();
    }
    HashMap::from([
        ("session_id".into(), session_id.into()),
        ("x-client-request-id".into(), session_id.into()),
        ("x-ms-client-request-id".into(), session_id.into()),
    ])
}

/// Strip Azure-specific tool_call cleanup fields.
pub fn strip_azure_tool_call_fields(tool_calls: &mut Vec<Value>) {
    for tc in tool_calls.iter_mut() {
        if let Some(obj) = tc.as_object_mut() {
            obj.remove("content_filter_results");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_normalize_reasoning() {
        let mut event = json!({
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "thinking..."}]
            }
        });
        normalize_azure_reasoning_event(&mut event);
        assert!(event.pointer("/item/summary").is_some());
        assert!(event.pointer("/item/content").is_none());
    }

    #[test]
    fn test_azure_session_headers() {
        let h = azure_session_headers("sess-123");
        assert_eq!(h.len(), 3);
        assert_eq!(h.get("session_id").unwrap(), "sess-123");
    }

    #[test]
    fn test_empty_session_id() {
        let h = azure_session_headers("");
        assert!(h.is_empty());
    }
}
