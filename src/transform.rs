//! Message transformation for cross-provider compatibility.
//!
//! Mirrors the Go `transform.go` which normalizes messages before sending
//! to different providers (image downgrade, thinking-to-text, etc.)

use crate::types::{ContentBlock, Message, Model, Role, StopReason};

const NON_VISION_USER_IMAGE_PLACEHOLDER: &str = "(image omitted: model does not support images)";
const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str = "(tool image omitted: model does not support images)";

/// Transform messages for a target model, handling cross-provider differences.
pub fn transform_messages(messages: &[Message], model: &Model) -> Vec<Message> {
    let (downgraded, _) = downgrade_unsupported_images(messages, model);
    let normalized = normalize_cross_model_content(downgraded, model);
    insert_synthetic_tool_results(normalized)
}

/// First pass: for assistant messages from a different provider/model, downgrade
/// thinking blocks to plain text, drop cross-model redacted thinking, and strip
/// cross-model tool-call thought signatures (mirrors transformMessages pass 1).
fn normalize_cross_model_content(messages: Vec<Message>, model: &Model) -> Vec<Message> {
    messages.into_iter().map(|msg| {
        if msg.role != Role::Assistant {
            return msg;
        }
        let is_same = msg.provider.as_deref() == Some(model.provider.as_str())
            && msg.api.as_deref() == Some(model.api.as_str())
            && msg.model.as_deref() == Some(model.id.as_str());
        let mut new_content: Vec<ContentBlock> = Vec::new();
        for block in &msg.content {
            match block {
                ContentBlock::Thinking { thinking, thinking_signature, redacted } => {
                    if *redacted {
                        // Opaque encrypted content; only valid for the same model.
                        if is_same { new_content.push(block.clone()); }
                    } else if is_same && thinking_signature.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
                        // Keep signed thinking for replay even if the text is empty.
                        new_content.push(block.clone());
                    } else if thinking.trim().is_empty() {
                        // Drop empty thinking.
                    } else if is_same {
                        new_content.push(block.clone());
                    } else {
                        // Cross-model: downgrade to plain text.
                        new_content.push(ContentBlock::Text { text: thinking.clone(), text_signature: None });
                    }
                }
                ContentBlock::Text { text, .. } => {
                    if is_same {
                        new_content.push(block.clone());
                    } else {
                        // Cross-model: strip the text signature.
                        new_content.push(ContentBlock::Text { text: text.clone(), text_signature: None });
                    }
                }
                ContentBlock::ToolCall { id, name, arguments, thought_signature } => {
                    if !is_same && thought_signature.is_some() {
                        new_content.push(ContentBlock::ToolCall {
                            id: id.clone(), name: name.clone(), arguments: arguments.clone(),
                            thought_signature: None,
                        });
                    } else {
                        new_content.push(block.clone());
                    }
                }
                other => new_content.push(other.clone()),
            }
        }
        Message { content: new_content, ..msg }
    }).collect()
}

/// Skip errored/aborted assistant turns and insert synthetic "No result provided"
/// tool results for orphaned tool calls (mirrors upstream transformMessages pass 2).
fn insert_synthetic_tool_results(messages: Vec<Message>) -> Vec<Message> {
    use std::collections::HashSet;
    let mut result: Vec<Message> = Vec::new();
    // Pending tool calls (id, name) from the most recent tool-call-bearing assistant.
    let mut pending: Vec<(String, String)> = Vec::new();
    let mut existing: HashSet<String> = HashSet::new();

    fn flush(result: &mut Vec<Message>, pending: &mut Vec<(String, String)>, existing: &mut HashSet<String>) {
        for (id, name) in pending.drain(..) {
            if !existing.contains(&id) {
                result.push(Message {
                    role: Role::ToolResult,
                    content: vec![ContentBlock::Text { text: "No result provided".to_string(), text_signature: None }],
                    timestamp: crate::utils::now_millis(),
                    api: None, provider: None, model: None, response_id: None,
                    response_model: None, diagnostics: Vec::new(), usage: None,
                    stop_reason: None, error_message: None,
                    tool_call_id: Some(id), tool_name: Some(name), is_error: true, details: None,
                });
            }
        }
        existing.clear();
    }

    for msg in messages {
        match msg.role {
            Role::Assistant => {
                flush(&mut result, &mut pending, &mut existing);
                // Skip errored/aborted assistant turns; they're incomplete and shouldn't replay.
                if matches!(msg.stop_reason, Some(StopReason::Error) | Some(StopReason::Aborted)) {
                    continue;
                }
                let tool_calls: Vec<(String, String)> = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::ToolCall { id, name, .. } => Some((id.clone(), name.clone())),
                    _ => None,
                }).collect();
                if !tool_calls.is_empty() {
                    pending = tool_calls;
                    existing.clear();
                }
                result.push(msg);
            }
            Role::ToolResult => {
                if let Some(ref id) = msg.tool_call_id {
                    existing.insert(id.clone());
                }
                result.push(msg);
            }
            Role::User => {
                flush(&mut result, &mut pending, &mut existing);
                result.push(msg);
            }
        }
    }
    flush(&mut result, &mut pending, &mut existing);
    result
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
            compat: Default::default(),
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

    fn msg(role: Role, content: Vec<ContentBlock>, stop: Option<StopReason>, tcid: Option<&str>) -> Message {
        Message {
            role, content, timestamp: 0,
            api: None, provider: None, model: None, response_id: None,
            response_model: None, diagnostics: Vec::new(), usage: None,
            stop_reason: stop, error_message: None,
            tool_call_id: tcid.map(|s| s.to_string()), tool_name: None, is_error: false, details: None,
        }
    }

    #[test]
    fn test_synthetic_tool_result_for_orphaned_call() {
        let messages = vec![
            msg(Role::Assistant, vec![ContentBlock::ToolCall {
                id: "tc1".into(), name: "search".into(),
                arguments: std::collections::HashMap::new(), thought_signature: None,
            }], Some(StopReason::ToolUse), None),
            // No tool result -> a synthetic one should be inserted before the next user turn.
            msg(Role::User, vec![ContentBlock::Text { text: "continue".into(), text_signature: None }], None, None),
        ];
        let result = transform_messages(&messages, &vision_model());
        // assistant, synthetic toolResult, user
        assert_eq!(result.len(), 3);
        assert_eq!(result[1].role, Role::ToolResult);
        assert_eq!(result[1].tool_call_id.as_deref(), Some("tc1"));
        assert!(result[1].is_error);
        assert!(matches!(&result[1].content[0], ContentBlock::Text { text, .. } if text == "No result provided"));
    }

    #[test]
    fn test_existing_tool_result_no_synthetic() {
        let messages = vec![
            msg(Role::Assistant, vec![ContentBlock::ToolCall {
                id: "tc1".into(), name: "s".into(),
                arguments: std::collections::HashMap::new(), thought_signature: None,
            }], Some(StopReason::ToolUse), None),
            msg(Role::ToolResult, vec![ContentBlock::Text { text: "ok".into(), text_signature: None }], None, Some("tc1")),
        ];
        let result = transform_messages(&messages, &vision_model());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_skip_errored_assistant_message() {
        let messages = vec![
            msg(Role::Assistant, vec![ContentBlock::Text { text: "partial".into(), text_signature: None }], Some(StopReason::Error), None),
            msg(Role::User, vec![ContentBlock::Text { text: "hi".into(), text_signature: None }], None, None),
        ];
        let result = transform_messages(&messages, &vision_model());
        // The errored assistant turn is dropped.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, Role::User);
    }

    #[test]
    fn test_cross_model_thinking_downgraded() {
        // Assistant message from a different model: thinking -> text, signatures stripped.
        let mut m = msg(Role::Assistant, vec![
            ContentBlock::Thinking { thinking: "reasoning".into(), thinking_signature: Some("sig".into()), redacted: false },
            ContentBlock::ToolCall { id: "tc1".into(), name: "s".into(), arguments: std::collections::HashMap::new(), thought_signature: Some("ts".into()) },
        ], Some(StopReason::ToolUse), None);
        m.provider = Some("anthropic".into());
        m.api = Some("anthropic-messages".into());
        m.model = Some("claude".into());
        let result = transform_messages(&[m], &vision_model());
        // gpt-4o (openai) target != anthropic source -> cross-model.
        assert!(matches!(&result[0].content[0], ContentBlock::Text { text, .. } if text == "reasoning"));
        assert!(matches!(&result[0].content[1], ContentBlock::ToolCall { thought_signature: None, .. }));
    }
}
