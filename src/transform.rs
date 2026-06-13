//! Message transformation for cross-provider compatibility.
//!
//! Mirrors the Go `transform.go` which normalizes messages before sending
//! to different providers (image downgrade, thinking-to-text, etc.)

use crate::types::{ContentBlock, Message, Model, Role};

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// Transform messages for a target model, handling cross-provider differences.
pub fn transform_messages(messages: &[Message], model: &Model) -> Vec<Message> {
    let (transformed, _) = downgrade_unsupported_images(messages, model);
    transformed
}

/// Replace consecutive image blocks with a single text placeholder, matching
/// upstream `replaceImagesWithPlaceholder`.
fn replace_images_with_placeholder(content: &[ContentBlock], placeholder: &str) -> (Vec<ContentBlock>, usize) {
    let mut result = Vec::with_capacity(content.len());
    let mut previous_was_placeholder = false;
    let mut downgrades = 0;
    for block in content {
        match block {
            ContentBlock::Image { .. } => {
                if !previous_was_placeholder {
                    result.push(ContentBlock::Text { text: placeholder.to_string(), text_signature: None });
                }
                downgrades += 1;
                previous_was_placeholder = true;
            }
            other => {
                previous_was_placeholder = matches!(other, ContentBlock::Text { text, .. } if text == placeholder);
                result.push(other.clone());
            }
        }
    }
    (result, downgrades)
}

/// Replace image content with text placeholders for non-vision models.
/// Only user and tool-result messages are downgraded (mirrors upstream).
fn downgrade_unsupported_images(messages: &[Message], model: &Model) -> (Vec<Message>, usize) {
    let supports_images = model.input.iter().any(|i| i == "image");
    if supports_images {
        return (messages.to_vec(), 0);
    }

    let mut downgrades = 0;
    let transformed: Vec<Message> = messages
        .iter()
        .map(|msg| {
            let placeholder = match msg.role {
                Role::User => NON_VISION_USER_IMAGE_PLACEHOLDER,
                Role::ToolResult => NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                Role::Assistant => return msg.clone(),
            };
            let (new_content, n) = replace_images_with_placeholder(&msg.content, placeholder);
            downgrades += n;
            Message { content: new_content, ..msg.clone() }
        })
        .collect();

    (transformed, downgrades)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelCost, Role};

    fn vision_model() -> Model {
        Model {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            api: "openai-completions".into(),
            provider: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            reasoning: false,
            thinking_level_map: None,
            input: vec!["text".into(), "image".into()],
            cost: ModelCost::default(),
            context_window: 128000,
            max_tokens: 4096,
            headers: None,
            api_key: None,
        }
    }

    fn text_only_model() -> Model {
        Model {
            input: vec!["text".into()],
            ..vision_model()
        }
    }

    #[test]
    fn test_preserves_images_for_vision_model() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text { text: "Look".into(), text_signature: None },
                ContentBlock::Image { data: "base64".into(), mime_type: "image/png".into() },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }];
        let result = transform_messages(&messages, &vision_model());
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(&result[0].content[1], ContentBlock::Image { .. }));
    }

    #[test]
    fn test_downgrades_images_for_text_model() {
        let messages = vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text { text: "Look".into(), text_signature: None },
                ContentBlock::Image { data: "base64".into(), mime_type: "image/png".into() },
            ],
            timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None, stop_reason: None, error_message: None,
            tool_call_id: None, tool_name: None, is_error: false,
            details: None,
        }];
        let result = transform_messages(&messages, &text_only_model());
        assert_eq!(result[0].content.len(), 2);
        assert!(matches!(&result[0].content[1], ContentBlock::Text { text, .. } if text.contains("omitted")));
    }
}
