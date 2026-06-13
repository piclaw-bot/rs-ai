//! Partial JSON parser for streaming tool call arguments.
//!
//! When providers stream tool call arguments as incremental JSON chunks,
//! we need to attempt parsing incomplete JSON by closing open structures and
//! repairing common malformations (raw control characters, invalid escapes).

use serde_json::Value;

/// Repair common JSON malformations inside string literals:
/// - escape raw control characters
/// - double backslashes before invalid escape characters
///
/// Mirrors upstream pi-ai `repairJson`.
pub fn repair_json(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let mut repaired = String::with_capacity(json.len());
    let mut in_string = false;
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];
        if !in_string {
            repaired.push(ch);
            if ch == '"' {
                in_string = true;
            }
            index += 1;
            continue;
        }
        if ch == '"' {
            repaired.push(ch);
            in_string = false;
            index += 1;
            continue;
        }
        if ch == '\\' {
            let next = chars.get(index + 1).copied();
            match next {
                None => {
                    repaired.push_str("\\\\");
                    index += 1;
                    continue;
                }
                Some('u') => {
                    let digits: String = chars.iter().skip(index + 2).take(4).collect();
                    if digits.len() == 4 && digits.chars().all(|c| c.is_ascii_hexdigit()) {
                        repaired.push_str("\\u");
                        repaired.push_str(&digits);
                        index += 6;
                        continue;
                    }
                    repaired.push_str("\\\\");
                    index += 1;
                    continue;
                }
                Some(n) if matches!(n, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') => {
                    repaired.push('\\');
                    repaired.push(n);
                    index += 2;
                    continue;
                }
                Some(_) => {
                    repaired.push_str("\\\\");
                    index += 1;
                    continue;
                }
            }
        }
        if is_control_character(ch) {
            repaired.push_str(&escape_control_character(ch));
        } else {
            repaired.push(ch);
        }
        index += 1;
    }

    repaired
}

fn is_control_character(ch: char) -> bool {
    (ch as u32) <= 0x1f
}

fn escape_control_character(ch: char) -> String {
    match ch {
        '\u{8}' => "\\b".to_string(),
        '\u{c}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        _ => format!("\\u{:04x}", ch as u32),
    }
}

/// Parse JSON, repairing common malformations on failure.
pub fn parse_json_with_repair(json: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(json) {
        return Some(v);
    }
    let repaired = repair_json(json);
    if repaired != json
        && let Ok(v) = serde_json::from_str::<Value>(&repaired) {
            return Some(v);
        }
    None
}

/// Parse a potentially incomplete streaming JSON string for tool-call arguments.
///
/// Always returns a value (an empty object on total failure), mirroring
/// upstream pi-ai `parseStreamingJson`.
pub fn parse_streaming_json(partial_json: &str) -> Value {
    if partial_json.trim().is_empty() {
        return serde_json::json!({});
    }
    if let Some(v) = parse_json_with_repair(partial_json) {
        return v;
    }
    if let Some(v) = parse_partial_json(partial_json) {
        return v;
    }
    if let Some(v) = parse_partial_json(&repair_json(partial_json)) {
        return v;
    }
    serde_json::json!({})
}

/// Attempt to parse potentially incomplete JSON by adding closing brackets/braces.
pub fn parse_partial_json(input: &str) -> Option<Value> {
    // Try direct parse first
    if let Ok(v) = serde_json::from_str(input) {
        return Some(v);
    }

    // Try closing open structures
    let mut fixed = input.to_string();
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for ch in input.chars() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => open_braces += 1,
            '}' => open_braces -= 1,
            '[' => open_brackets += 1,
            ']' => open_brackets -= 1,
            _ => {}
        }
    }

    // If we're in a string, close it
    if in_string {
        fixed.push('"');
    }

    // Close open structures
    for _ in 0..open_brackets {
        fixed.push(']');
    }
    for _ in 0..open_braces {
        fixed.push('}');
    }

    serde_json::from_str(&fixed).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_json() {
        let v = parse_partial_json(r#"{"key": "value"}"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["key"], "value");
    }

    #[test]
    fn test_partial_object() {
        let v = parse_partial_json(r#"{"key": "val"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["key"], "val");
    }

    #[test]
    fn test_partial_array() {
        let v = parse_partial_json(r#"[1, 2"#);
        assert!(v.is_some());
        let arr = v.unwrap();
        assert_eq!(arr.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_nested_partial() {
        let v = parse_partial_json(r#"{"a": {"b": [1, 2"#);
        assert!(v.is_some());
    }

    #[test]
    fn test_empty_returns_none() {
        assert!(parse_partial_json("").is_none());
    }

    #[test]
    fn test_repair_control_characters() {
        // Raw newline inside a string is invalid JSON; repair should fix it.
        let raw = "{\"text\": \"line1\nline2\"}";
        assert!(serde_json::from_str::<Value>(raw).is_err());
        let v = parse_streaming_json(raw);
        assert_eq!(v["text"], "line1\nline2");
    }

    #[test]
    fn test_repair_invalid_escape() {
        // Invalid escape \x should be repaired by doubling the backslash.
        let raw = r#"{"path": "C:\Users"}"#;
        let v = parse_streaming_json(raw);
        assert_eq!(v["path"], r"C:\Users");
    }

    #[test]
    fn test_streaming_json_empty_is_object() {
        assert_eq!(parse_streaming_json(""), serde_json::json!({}));
        assert_eq!(parse_streaming_json("   "), serde_json::json!({}));
    }

    #[test]
    fn test_streaming_json_partial_recovers() {
        let v = parse_streaming_json(r#"{"q": "rust"#);
        assert_eq!(v["q"], "rust");
    }
}
