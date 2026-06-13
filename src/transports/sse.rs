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
    let mut parser = SseParser::default();
    let mut events = Vec::new();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                events.push(SseEvent {
                    event: EVENT_ERROR.to_string(),
                    data: e.to_string(),
                    ..Default::default()
                });
                return events;
            }
        };
        events.extend(parser.process_line(&line));
    }

    if let Some(ev) = parser.finish() {
        events.push(ev);
    }

    events
}

#[derive(Debug, Default, Clone)]
pub struct SseParser {
    line_buffer: String,
    event_type: String,
    data_lines: Vec<String>,
    last_id: String,
    last_retry: Option<u32>,
    current_id: Option<String>,
    current_retry: Option<u32>,
}

impl SseParser {
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.line_buffer.push_str(chunk);
        let mut events = Vec::new();

        while let Some(pos) = self.line_buffer.find('\n') {
            let mut line = self.line_buffer[..pos].to_string();
            self.line_buffer.drain(..=pos);
            if line.ends_with('\r') {
                line.pop();
            }
            events.extend(self.process_line(&line));
        }

        events
    }

    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.line_buffer.is_empty() {
            let line = std::mem::take(&mut self.line_buffer);
            if let Some(ev) = self.process_line(line.trim_end_matches('\r')).into_iter().next() {
                return Some(ev);
            }
        }
        self.dispatch_event()
    }

    fn process_line(&mut self, line: &str) -> Vec<SseEvent> {
        if line.is_empty() {
            return self.dispatch_event().into_iter().collect();
        }

        if line.starts_with(':') {
            return Vec::new();
        }

        let (field, value) = match line.find(':') {
            Some(pos) => {
                let f = &line[..pos];
                let v = line[pos + 1..].strip_prefix(' ').unwrap_or(&line[pos + 1..]);
                (f, v.to_string())
            }
            None => (line, String::new()),
        };

        match field {
            "event" => self.event_type = value,
            "data" => self.data_lines.push(value),
            "id" => {
                self.current_id = Some(value.clone());
                self.last_id = value;
            }
            "retry" => {
                if let Ok(n) = value.parse::<u32>() {
                    self.current_retry = Some(n);
                    self.last_retry = Some(n);
                }
            }
            _ => {}
        }

        Vec::new()
    }

    fn dispatch_event(&mut self) -> Option<SseEvent> {
        if self.data_lines.is_empty() {
            self.event_type.clear();
            self.current_id = None;
            self.current_retry = None;
            return None;
        }

        let ev = SseEvent {
            event: if self.event_type.is_empty() {
                "message".to_string()
            } else {
                self.event_type.clone()
            },
            data: self.data_lines.join("\n"),
            id: self.current_id.clone().unwrap_or_else(|| self.last_id.clone()),
            retry: self.current_retry.or(self.last_retry),
        };

        self.event_type.clear();
        self.data_lines.clear();
        self.current_id = None;
        self.current_retry = None;
        Some(ev)
    }
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

    #[test]
    fn test_incremental_feed() {
        let mut parser = SseParser::default();
        let mut events = Vec::new();
        events.extend(parser.feed("data: hel"));
        assert!(events.is_empty());
        events.extend(parser.feed("lo\n\n"));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "hello");
    }
}
