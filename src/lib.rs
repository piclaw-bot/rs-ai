//! rs-ai — Unified LLM API with automatic model discovery, streaming,
//! tool calling, and multi-provider support.
//!
//! A Rust port of [@earendil-works/pi-ai](https://www.npmjs.com/package/@earendil-works/pi-ai).
//!
//! # Quick start
//!
//! ```no_run
//! use rs_ai::{registry, provider_id};
//!
//! #[tokio::main]
//! async fn main() {
//!     registry::register_builtin_models();
//!     let model = registry::get_model(provider_id::OPENAI, "gpt-4o-mini").unwrap();
//!     // ... stream or complete
//! }
//! ```

pub mod types;
pub mod events;
pub mod registry;
pub mod env;
pub mod compat;
pub mod provider;
pub mod transports;
pub mod images;
pub mod models_generated;
pub mod transform;
pub mod simple_options;
pub mod retry;
pub mod logger;
pub mod jsonparse;
pub mod harness;
pub mod utils;
pub mod context;
pub mod diagnostics;
pub mod azure;
pub mod session_resources;
pub mod prompt_cache;
pub mod validation;
pub mod oauth;

// Re-exports for convenience
pub use types::*;
pub use types::provider_id;
pub use events::*;
pub use registry::{stream, complete};

#[cfg(test)]
mod registry_test;
#[cfg(test)]
mod env_test;
