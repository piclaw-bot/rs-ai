//! Image generation types.

use serde::{Deserialize, Serialize};
use crate::types::{ModelCost, StopReason, Usage};

/// Image API identifier.
pub type ImagesApi = String;

/// Image provider identifier.
pub type ImagesProvider = String;

/// Input content for image generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageInput {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

/// Image generation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesContext {
    pub input: Vec<ImageInput>,
}

/// Output item from image generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ImageOutput {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
}

/// Image model definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImagesModel {
    pub id: String,
    pub name: String,
    pub api: ImagesApi,
    pub provider: ImagesProvider,
    pub base_url: String,
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
    pub cost: ModelCost,
}

/// Result of an image generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantImages {
    pub api: ImagesApi,
    pub provider: ImagesProvider,
    pub model: String,
    pub output: Vec<ImageOutput>,
    pub stop_reason: StopReason,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}
