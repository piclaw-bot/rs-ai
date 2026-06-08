//! Provider implementations (blank-import to register).
//!
//! Each submodule registers its provider on first use via `register()`.

pub mod openai;
pub mod anthropic;
pub mod google;
pub mod mistral;
pub mod responses;
pub mod faux;
pub mod bedrock;
pub mod codex;
pub mod geminicli;
