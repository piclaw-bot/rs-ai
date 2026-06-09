//! Provider and model registries, plus top-level stream/complete API.

use std::collections::HashMap;
use std::sync::RwLock;

use std::sync::Arc;
use crate::events::Event;
use crate::types::{Api, Context, Model, StreamOptions, StopReason, Message};

use tokio_stream::Stream;

/// Trait that provider implementations must satisfy.
pub trait ApiProvider: Send + Sync {
    /// The wire protocol this provider handles.
    fn api(&self) -> &str;

    /// Start a streaming request; returns a boxed async Stream of events.
    ///
    /// The returned stream may borrow the request inputs, but not the provider
    /// instance itself. Provider adapters in rs-ai are stateless.
    fn stream<'a>(
        &self,
        model: &'a Model,
        context: &'a Context,
        opts: &'a StreamOptions,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>>;
}

// --- Global registries ---

static API_PROVIDERS: std::sync::LazyLock<RwLock<HashMap<Api, Arc<dyn ApiProvider>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

static MODELS: std::sync::LazyLock<RwLock<HashMap<String, Model>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a provider implementation.
pub fn register_api(provider: Arc<dyn ApiProvider>) {
    let api = provider.api().to_string();
    API_PROVIDERS.write().unwrap().insert(api, provider);
}

/// Retrieve a registered provider by API name.
pub fn get_api_provider(api: &str) -> bool {
    API_PROVIDERS.read().unwrap().contains_key(api)
}

/// Register a model in the global registry.
pub fn register_model(model: Model) {
    let key = format!("{}/{}", model.provider, model.id);
    MODELS.write().unwrap().insert(key, model);
}

/// Look up a model by provider and ID.
pub fn get_model(provider: &str, id: &str) -> Option<Model> {
    let key = format!("{}/{}", provider, id);
    MODELS.read().unwrap().get(&key).cloned()
}

/// List all models, optionally filtered by provider.
pub fn list_models(provider: Option<&str>) -> Vec<Model> {
    MODELS
        .read()
        .unwrap()
        .values()
        .filter(|m| provider.is_none_or(|p| m.provider == p))
        .cloned()
        .collect()
}

/// List all registered provider names.
pub fn list_providers() -> Vec<String> {
    let models = MODELS.read().unwrap();
    let mut seen: Vec<String> = models
        .values()
        .map(|m| m.provider.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    seen.sort();
    seen
}

/// Register all built-in models from the generated registry.
pub fn register_builtin_models() {
    for model in crate::models_generated::builtin_models() {
        register_model(model);
    }
}

/// Start a streaming LLM request.
pub fn stream<'a>(
    model: &'a Model,
    context: &'a Context,
    opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn Stream<Item = Event> + Send + 'a>> {
    let provider = {
        let providers = API_PROVIDERS.read().unwrap();
        providers.get(&model.api).cloned()
    };
    match provider {
        Some(provider) => provider.stream(model, context, opts),
        None => {
            let err = Event::Error {
                reason: StopReason::Error,
                error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(
                    format!("no provider registered for API {:?}", model.api),
                )),
                message: None,
            };
            Box::pin(tokio_stream::once(err))
        }
    }
}

/// Non-streaming completion (collects stream to final message).
pub async fn complete(
    model: &Model,
    context: &Context,
    opts: &StreamOptions,
) -> Result<Message, Arc<dyn std::error::Error + Send + Sync>> {
    use tokio_stream::StreamExt;

    let mut events = stream(model, context, opts);
    let mut result: Option<Message> = None;
    let mut last_err: Option<Arc<dyn std::error::Error + Send + Sync>> = None;

    while let Some(event) = events.next().await {
        match event {
            Event::Done { message, .. } => {
                result = Some(message);
            }
            Event::Error { error, message, .. } => {
                last_err = Some(error);
                if let Some(msg) = message {
                    result = Some(msg);
                }
            }
            _ => {}
        }
    }

    if let Some(err) = last_err {
        Err(err)
    } else {
        result.ok_or_else(|| Arc::from(Box::<dyn std::error::Error + Send + Sync>::from("stream ended without done or error event")))
    }
}

/// Unregister a provider by API name.
pub fn unregister_api(api: &str) {
    API_PROVIDERS.write().unwrap().remove(api);
}

/// Clear all registered providers.
pub fn clear_api_providers() {
    API_PROVIDERS.write().unwrap().clear();
}

/// Clear all registered models.
pub fn clear_models() {
    MODELS.write().unwrap().clear();
}
