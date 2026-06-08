//! Session resource management and cleanup.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Tracks session resources (e.g., cached WebSocket connections) for cleanup.
pub struct SessionResources {
    resources: Arc<Mutex<HashMap<String, SessionEntry>>>,
}

struct SessionEntry {
    session_id: String,
    provider: String,
    created_at: i64,
}

impl SessionResources {
    pub fn new() -> Self {
        Self {
            resources: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a session resource for future cleanup.
    pub fn register(&self, session_id: &str, provider: &str) {
        let mut map = self.resources.lock().unwrap();
        map.insert(session_id.to_string(), SessionEntry {
            session_id: session_id.to_string(),
            provider: provider.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
        });
    }

    /// Remove a session resource.
    pub fn unregister(&self, session_id: &str) {
        self.resources.lock().unwrap().remove(session_id);
    }

    /// List all registered session IDs.
    pub fn list(&self) -> Vec<String> {
        self.resources.lock().unwrap().keys().cloned().collect()
    }

    /// Cleanup all registered session resources.
    pub fn cleanup_all(&self) -> usize {
        let mut map = self.resources.lock().unwrap();
        let count = map.len();
        map.clear();
        count
    }
}

impl Default for SessionResources {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let sr = SessionResources::new();
        sr.register("sess-1", "openai-codex");
        sr.register("sess-2", "openai-codex");
        assert_eq!(sr.list().len(), 2);
        sr.unregister("sess-1");
        assert_eq!(sr.list().len(), 1);
        assert_eq!(sr.cleanup_all(), 1);
        assert_eq!(sr.list().len(), 0);
    }
}
