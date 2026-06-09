//! Image generation API surface.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

pub mod types;
pub mod openrouter;
pub mod models_generated;

pub use types::*;

/// Trait for image API providers.
pub trait ImagesApiProvider: Send + Sync {
    fn api(&self) -> &str;
    fn generate<'a>(
        &self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        opts: &'a openrouter::ImagesOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AssistantImages> + Send + 'a>>;
}

static IMAGE_API_PROVIDERS: LazyLock<RwLock<HashMap<String, Arc<dyn ImagesApiProvider>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static IMAGE_MODELS: LazyLock<RwLock<HashMap<String, ImagesModel>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register an image API provider.
pub fn register_images_api_provider(provider: Arc<dyn ImagesApiProvider>) {
    let api = provider.api().to_string();
    IMAGE_API_PROVIDERS.write().unwrap().insert(api, provider);
}

/// Unregister an image API provider.
pub fn unregister_images_api_provider(api: &str) {
    IMAGE_API_PROVIDERS.write().unwrap().remove(api);
}

/// Clear all image API providers.
pub fn clear_images_api_providers() {
    IMAGE_API_PROVIDERS.write().unwrap().clear();
}

/// Register an image model.
pub fn register_image_model(model: ImagesModel) {
    let key = format!("{}/{}", model.provider, model.id);
    IMAGE_MODELS.write().unwrap().insert(key, model);
}

/// Register all built-in image models.
pub fn register_builtin_image_models() {
    for model in models_generated::builtin_image_models() {
        register_image_model(model);
    }
}

/// Clear all image models.
pub fn clear_image_models() {
    IMAGE_MODELS.write().unwrap().clear();
}

/// Get an image model by provider and ID.
pub fn get_image_model(provider: &str, id: &str) -> Option<ImagesModel> {
    let key = format!("{}/{}", provider, id);
    IMAGE_MODELS.read().unwrap().get(&key).cloned()
}

/// List image models, optionally filtered by provider.
pub fn list_image_models(provider: Option<&str>) -> Vec<ImagesModel> {
    IMAGE_MODELS.read().unwrap().values()
        .filter(|m| provider.is_none_or(|p| m.provider == p))
        .cloned()
        .collect()
}

/// List image providers.
pub fn list_image_providers() -> Vec<String> {
    let models = IMAGE_MODELS.read().unwrap();
    let mut seen: Vec<String> = models.values().map(|m| m.provider.clone()).collect::<std::collections::HashSet<_>>().into_iter().collect();
    seen.sort();
    seen
}

/// Generate images using the registered provider for a model.
pub async fn generate_images(
    model: &ImagesModel,
    context: &ImagesContext,
    opts: &openrouter::ImagesOptions,
) -> AssistantImages {
    let provider = {
        let providers = IMAGE_API_PROVIDERS.read().unwrap();
        providers.get(&model.api).cloned()
    };
    match provider {
        Some(p) => p.generate(model, context, opts).await,
        None => AssistantImages {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            output: Vec::new(),
            stop_reason: crate::types::StopReason::Error,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64,
            response_id: None,
            usage: None,
            error_message: Some(format!("no image provider registered for API {}", model.api)),
        },
    }
}

struct OpenRouterImagesProvider;
impl ImagesApiProvider for OpenRouterImagesProvider {
    fn api(&self) -> &str { "openrouter-images" }
    fn generate<'a>(
        &self,
        model: &'a ImagesModel,
        context: &'a ImagesContext,
        opts: &'a openrouter::ImagesOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AssistantImages> + Send + 'a>> {
        Box::pin(openrouter::generate_openrouter(model, context, opts))
    }
}

/// Register all built-in image providers.
pub fn register_builtin_image_providers() {
    register_images_api_provider(Arc::new(OpenRouterImagesProvider));
}
