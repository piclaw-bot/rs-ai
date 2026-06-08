//! SSE (Server-Sent Events) parser for streaming providers.

use std::io::BufRead;

/// A single SSE event.
#[derive(Debug, Clone, Default)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: String,
    pub retry: Option<u32>,
}

pub const EVENT_ERROR: &str = "__error__";

/// Parse SSE events from a buffered reader.
///
/// Yields events as they are dispatched (on empty lines).
/// Implements sticky `id` and `retry` fields per the SSE spec.
pub fn parse<R: BufRead>(reader: R) -> Vec<SseEvent> {
    let mut events = Vec::new();
    let mut event_type = String::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut last_id = String::new();
    let mut last_retry: Option<u32> = None;
    let mut current_id: Option<String> = None;
    let mut current_retry: Option<u32> = None;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                events.push(SseEvent {
                    event: EVENT_ERROR.to_string(),
                    data: e.to_string(),
                    ..Default::default()
                });
                break;
            }
        };

        if line.is_empty() {
            if !data_lines.is_empty() {
                let ev = SseEvent {
                    event: if event_type.is_empty() {
                        "message".to_string()
                    } else {
                        event_type.clone()
                    },
                    data: data_lines.join("\n"),
                    id: current_id.clone().unwrap_or_else(|| last_id.clone()),
                    retry: current_retry.or(last_retry),
                };
                events.push(ev);
            }
            // Reset per-event state; sticky id/retry persist
            event_type.clear();
            data_lines.clear();
            current_id = None;
            current_retry = None;
            continue;
        }

        if line.starts_with(':') {
            continue; // comment
        }

        let (field, value) = match line.find(':') {
            Some(pos) => {
                let f = &line[..pos];
                let v = line[pos + 1..].strip_prefix(' ').unwrap_or(&line[pos + 1..]);
                (f, v.to_string())
            }
            None => (line.as_str(), String::new()),
        };

        match field {
            "event" => event_type = value,
            "data" => data_lines.push(value),
            "id" => {
                current_id = Some(value.clone());
                last_id = value;
            }
            "retry" => {
                if let Ok(n) = value.parse::<u32>() {
                    current_retry = Some(n);
                    last_retry = Some(n);
                }
            }
            _ => {}
        }
    }

    // Flush last event if no trailing blank line
    if !data_lines.is_empty() {
        events.push(SseEvent {
            event: if event_type.is_empty() {
                "message".to_string()
            } else {
                event_type
            },
            data: data_lines.join("\n"),
            id: current_id.unwrap_or(last_id),
            retry: current_retry.or(last_retry),
        });
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parse() {
        let input = "event: message_start\ndata: {\"type\":\"start\"}\n\ndata: hello\n\n";
        let events = parse(input.as_bytes());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event, "message_start");
        assert_eq!(events[1].event, "message");
        assert_eq!(events[1].data, "hello");
    }

    #[test]
    fn test_sticky_id() {
        let input = "id: 42\ndata: first\n\ndata: second\n\n";
        let events = parse(input.as_bytes());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, "42");
        assert_eq!(events[1].id, "42"); // sticky
    }

    #[test]
    fn test_multiline_data() {
        let input = "data: line1\ndata: line2\n\n";
        let events = parse(input.as_bytes());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }
}
