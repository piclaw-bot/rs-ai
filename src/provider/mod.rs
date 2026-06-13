//! Provider implementations and registration helpers.

use std::sync::Arc;

use crate::registry::{self, ApiProvider};
use crate::types::{Context, Model, StreamOptions};
use crate::events::Event;
use tokio_stream::Stream;

pub mod openai;
pub mod anthropic;
pub mod google;
pub mod mistral;
pub mod responses;
pub mod faux;
pub mod bedrock;
pub mod codex;
pub mod geminicli;

struct OpenAIProvider;
impl ApiProvider for OpenAIProvider {
    fn api(&self) -> &str { "openai-completions" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        openai::stream_openai(model, context, opts)
    }
}

struct OpenAIResponsesProvider;
impl ApiProvider for OpenAIResponsesProvider {
    fn api(&self) -> &str { "openai-responses" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        responses::stream_responses(model, context, opts)
    }
}

struct AzureOpenAIResponsesProvider;
impl ApiProvider for AzureOpenAIResponsesProvider {
    fn api(&self) -> &str { "azure-openai-responses" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        responses::stream_azure_responses(model, context, opts)
    }
}

struct AnthropicProvider;
impl ApiProvider for AnthropicProvider {
    fn api(&self) -> &str { "anthropic-messages" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        anthropic::stream_anthropic(model, context, opts)
    }
}

struct GoogleProvider;
impl ApiProvider for GoogleProvider {
    fn api(&self) -> &str { "google-generative-ai" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        google::stream_google(model, context, opts)
    }
}

struct GoogleVertexProvider;
impl ApiProvider for GoogleVertexProvider {
    fn api(&self) -> &str { "google-vertex" }
    // NOTE: Vertex AI responses use the same @google/genai format as Gemini, so the
    // shared `stream_google` decoder is correct. However, production Vertex auth
    // requires GCP Application Default Credentials / service-account token exchange
    // and a project/location-scoped endpoint (the model `base_url` carries a
    // `{location}` sentinel the upstream SDK resolves internally). That auth flow is
    // a known limitation here: only Vertex API-key access via the shared path works;
    // full ADC support would require a GCP auth dependency.
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        google::stream_google(model, context, opts)
    }
}

struct GeminiCliProvider;
impl ApiProvider for GeminiCliProvider {
    fn api(&self) -> &str { "google-gemini-cli" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        geminicli::stream_geminicli(model, context, opts)
    }
}

struct MistralProvider;
impl ApiProvider for MistralProvider {
    fn api(&self) -> &str { "mistral-conversations" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        mistral::stream_mistral(model, context, opts)
    }
}

struct BedrockProvider;
impl ApiProvider for BedrockProvider {
    fn api(&self) -> &str { "bedrock-converse-stream" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        bedrock::stream_bedrock(model, context, opts)
    }
}

struct CodexProvider;
impl ApiProvider for CodexProvider {
    fn api(&self) -> &str { "openai-codex-responses" }
    fn stream<'a>(&self, model: &'a Model, context: &'a Context, opts: &'a StreamOptions) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
        codex::stream_codex(model, context, opts)
    }
}

/// Register all built-in provider implementations.
pub fn register_builtin_providers() {
    registry::register_api(Arc::new(OpenAIProvider));
    registry::register_api(Arc::new(OpenAIResponsesProvider));
    registry::register_api(Arc::new(AzureOpenAIResponsesProvider));
    registry::register_api(Arc::new(AnthropicProvider));
    registry::register_api(Arc::new(GoogleProvider));
    registry::register_api(Arc::new(GoogleVertexProvider));
    registry::register_api(Arc::new(GeminiCliProvider));
    registry::register_api(Arc::new(MistralProvider));
    registry::register_api(Arc::new(BedrockProvider));
    registry::register_api(Arc::new(CodexProvider));
}
