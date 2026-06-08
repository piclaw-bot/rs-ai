//! Partial JSON parser for streaming tool call arguments.
//!
//! When providers stream tool call arguments as incremental JSON chunks,
//! we need to attempt parsing incomplete JSON by closing open structures.

use serde_json::Value;

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
}
