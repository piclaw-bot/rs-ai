//! Amazon Bedrock ConverseStream provider.

use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContent, ConversationRole, Message as BedrockMessage,
    SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolResultStatus, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_types::{Document, Number};

use crate::events::Event;
use crate::types::*;

/// Convert a serde_json::Value into an AWS smithy Document.
fn json_to_document(v: &serde_json::Value) -> Document {
    match v {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(Number::NegInt(i))
            } else {
                Document::Number(Number::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(a) => Document::Array(a.iter().map(json_to_document).collect()),
        serde_json::Value::Object(o) => {
            Document::Object(o.iter().map(|(k, v)| (k.clone(), json_to_document(v))).collect())
        }
    }
}

/// Start a Bedrock ConverseStream.
pub fn stream_bedrock<'a>(
    model: &'a Model,
    context: &'a Context,
    opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    Box::pin(async_stream::stream! {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let client = BedrockClient::new(&config);

        let mut messages = Vec::new();
        for msg in &context.messages {
            let role = match msg.role {
                Role::User | Role::ToolResult => ConversationRole::User,
                Role::Assistant => ConversationRole::Assistant,
            };
            let mut content: Vec<BedrockContent> = Vec::new();
            if msg.role == Role::ToolResult {
                let text = msg.content.iter().filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                let status = if msg.is_error { ToolResultStatus::Error } else { ToolResultStatus::Success };
                if let Ok(trb) = ToolResultBlock::builder()
                    .tool_use_id(msg.tool_call_id.clone().unwrap_or_default())
                    .content(ToolResultContentBlock::Text(text))
                    .status(status)
                    .build()
                {
                    content.push(BedrockContent::ToolResult(trb));
                }
            } else {
                for b in &msg.content {
                    match b {
                        ContentBlock::Text { text, .. } => {
                            content.push(BedrockContent::Text(text.clone()));
                        }
                        ContentBlock::ToolCall { id, name, arguments, .. } => {
                            let args_value = serde_json::to_value(arguments).unwrap_or_else(|_| serde_json::json!({}));
                            if let Ok(tub) = ToolUseBlock::builder()
                                .tool_use_id(id.clone())
                                .name(name.clone())
                                .input(json_to_document(&args_value))
                                .build()
                            {
                                content.push(BedrockContent::ToolUse(tub));
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !content.is_empty() {
                messages.push(
                    BedrockMessage::builder()
                        .role(role)
                        .set_content(Some(content))
                        .build()
                        .unwrap()
                );
            }
        }

        let mut req = client
            .converse_stream()
            .model_id(&model.id)
            .set_messages(Some(messages));

        if let Some(ref prompt) = context.system_prompt {
            req = req.system(SystemContentBlock::Text(prompt.clone()));
        }

        if !context.tools.is_empty() {
            let mut tool_list = Vec::new();
            for t in &context.tools {
                if let Ok(spec) = ToolSpecification::builder()
                    .name(t.name.clone())
                    .description(t.description.clone())
                    .input_schema(ToolInputSchema::Json(json_to_document(&t.parameters)))
                    .build()
                {
                    tool_list.push(Tool::ToolSpec(spec));
                }
            }
            if let Ok(tc) = ToolConfiguration::builder().set_tools(Some(tool_list)).build() {
                req = req.tool_config(tc);
            }
        }

        // Enable thinking for Anthropic Claude models on Bedrock (additionalModelRequestFields).
        if model.reasoning
            && (model.id.contains("claude") || model.id.contains("anthropic"))
            && let Some(level) = opts.reasoning.as_ref() {
            let key = format!("{:?}", level).to_lowercase();
            let default_budget = match key.as_str() {
                "minimal" => 1024, "low" => 2048, "medium" => 8192, _ => 16384,
            };
            let budget = opts.thinking_budgets.as_ref().and_then(|b| match key.as_str() {
                "minimal" => b.minimal, "low" => b.low, "medium" => b.medium, _ => b.high,
            }).unwrap_or(default_budget);
            let fields = serde_json::json!({
                "thinking": { "type": "enabled", "budget_tokens": budget, "display": "summarized" },
                "anthropic_beta": ["interleaved-thinking-2025-05-14"],
            });
            req = req.additional_model_request_fields(json_to_document(&fields));
        }

        let result = req.send().await;

        let output = match result {
            Ok(o) => o,
            Err(e) => {
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(format_bedrock_error(&e.to_string()))),
                    message: None,
                };
                return;
            }
        };

        let mut partial = Message {
            role: Role::Assistant,
            content: Vec::new(),
            timestamp: crate::utils::now_millis(),
            api: Some(model.api.clone()),
            provider: Some(model.provider.clone()),
            model: Some(model.id.clone()),
            response_id: None,
            response_model: None,
            diagnostics: Vec::new(),
            usage: None,
            stop_reason: None,
            error_message: None,
            tool_call_id: None,
            tool_name: None,
            is_error: false,
            details: None,
        };

        yield Event::Start { partial: partial.clone() };

        let mut current_text = String::new();
        let mut text_started = false;
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_args = String::new();
        let mut in_tool_block = false;
        let mut current_thinking = String::new();
        let mut current_thinking_signature: Option<String> = None;
        let mut thinking_started = false;

        let mut recv = output.stream;
        loop {
            match recv.recv().await {
                Ok(Some(event)) => {
                    use aws_sdk_bedrockruntime::types::ConverseStreamOutput;
                    match event {
                        ConverseStreamOutput::ContentBlockStart(start) => {
                            if let Some(aws_sdk_bedrockruntime::types::ContentBlockStart::ToolUse(tu)) = start.start() {
                                in_tool_block = true;
                                current_tool_id = tu.tool_use_id().to_string();
                                current_tool_name = tu.name().to_string();
                                current_tool_args.clear();
                                yield Event::ToolCallStart { id: current_tool_id.clone(), name: current_tool_name.clone() };
                            } else if !text_started {
                                text_started = true;
                                yield Event::TextStart;
                            }
                        }
                        ConverseStreamOutput::ContentBlockDelta(delta) => {
                            if let Some(d) = delta.delta() {
                                match d {
                                    aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(t) => {
                                        current_text.push_str(t);
                                        yield Event::TextDelta { delta: t.to_string() };
                                    }
                                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ToolUse(tu) => {
                                        let input = tu.input();
                                        current_tool_args.push_str(input);
                                        yield Event::ToolCallDelta { delta: input.to_string() };
                                    }
                                    aws_sdk_bedrockruntime::types::ContentBlockDelta::ReasoningContent(rc) => {
                                        use aws_sdk_bedrockruntime::types::ReasoningContentBlockDelta;
                                        match rc {
                                            ReasoningContentBlockDelta::Text(t) => {
                                                if !thinking_started {
                                                    thinking_started = true;
                                                    yield Event::ThinkingStart;
                                                }
                                                current_thinking.push_str(t);
                                                yield Event::ThinkingDelta { delta: t.to_string() };
                                            }
                                            ReasoningContentBlockDelta::Signature(s) => {
                                                current_thinking_signature = Some(s.to_string());
                                            }
                                            _ => {}
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        ConverseStreamOutput::ContentBlockStop(_) => {
                            if in_tool_block {
                                in_tool_block = false;
                                let parsed = crate::jsonparse::parse_streaming_json(&current_tool_args);
                                let arguments = match &parsed {
                                    serde_json::Value::Object(map) => map.clone().into_iter().collect(),
                                    _ => std::collections::HashMap::new(),
                                };
                                partial.content.push(ContentBlock::ToolCall {
                                    id: current_tool_id.clone(),
                                    name: current_tool_name.clone(),
                                    arguments,
                                    thought_signature: None,
                                });
                                yield Event::ToolCallEnd {
                                    id: std::mem::take(&mut current_tool_id),
                                    name: std::mem::take(&mut current_tool_name),
                                    arguments: parsed,
                                };
                                current_tool_args.clear();
                            } else if thinking_started {
                                thinking_started = false;
                                partial.content.push(ContentBlock::Thinking {
                                    thinking: std::mem::take(&mut current_thinking),
                                    thinking_signature: current_thinking_signature.take(),
                                    redacted: false,
                                });
                                yield Event::ThinkingEnd;
                            } else if text_started {
                                text_started = false;
                                yield Event::TextEnd;
                            }
                        }
                        ConverseStreamOutput::MessageStop(stop) => {
                            use aws_sdk_bedrockruntime::types::StopReason as BedrockStop;
                            let reason = stop.stop_reason();
                            partial.stop_reason = Some(match reason {
                                BedrockStop::EndTurn | BedrockStop::StopSequence => StopReason::Stop,
                                BedrockStop::MaxTokens | BedrockStop::ModelContextWindowExceeded => StopReason::Length,
                                BedrockStop::ToolUse => StopReason::ToolUse,
                                // content_filtered, guardrail_intervened, malformed_tool_use, etc.
                                other => {
                                    partial.error_message = Some(format!("Bedrock stop reason: {}", other));
                                    StopReason::Error
                                }
                            });
                        }
                        ConverseStreamOutput::Metadata(meta) => {
                            if let Some(u) = meta.usage() {
                                partial.usage = Some(Usage {
                                    input: u.input_tokens() as u32,
                                    output: u.output_tokens() as u32,
                                    cache_read: u.cache_read_input_tokens().unwrap_or(0) as u32,
                                    cache_write: u.cache_write_input_tokens().unwrap_or(0) as u32,
                                    total_tokens: u.total_tokens() as u32,
                                    ..Default::default()
                                });
                            }
                        }
                        _ => {}
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    yield Event::Error {
                        reason: StopReason::Error,
                        error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(format_bedrock_error(&e.to_string()))),
                        message: Some(partial.clone()),
                    };
                    return;
                }
            }
        }

        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text { text: current_text, text_signature: None });
        }
        if let Some(ref mut u) = partial.usage {
            crate::simple_options::finalize_usage(model, u);
        }
        match partial.stop_reason.clone() {
            Some(StopReason::Error) => {
                let msg = partial.error_message.clone().unwrap_or_else(|| "Provider returned an error stop reason".to_string());
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(msg)),
                    message: Some(partial),
                };
            }
            Some(reason) => {
                yield Event::Done { reason, message: partial };
            }
            None => {
                yield Event::Done { reason: StopReason::Stop, message: partial };
            }
        }
    })
}

/// AWS docs explaining how to configure a supported Bedrock data retention mode.
const BEDROCK_DATA_RETENTION_DOCS_URL: &str =
    "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html";

/// Append a data-retention docs hint when the error references retention mode
/// (mirrors upstream pi-ai formatBedrockError).
pub(crate) fn format_bedrock_error(message: &str) -> String {
    if message.to_lowercase().contains("data retention mode") {
        format!("{} See {} for supported data retention modes.", message, BEDROCK_DATA_RETENTION_DOCS_URL)
    } else {
        message.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{format_bedrock_error, json_to_document};
    use aws_smithy_types::{Document, Number};

    #[test]
    fn test_json_to_document_roundtrip_shapes() {
        let v = serde_json::json!({"q": "rust", "n": 3, "f": 1.5, "b": true, "arr": [1, 2], "nil": null});
        let doc = json_to_document(&v);
        let obj = doc.as_object().expect("object");
        assert!(matches!(obj.get("q"), Some(Document::String(s)) if s == "rust"));
        assert!(matches!(obj.get("n"), Some(Document::Number(Number::PosInt(3)))));
        assert!(matches!(obj.get("b"), Some(Document::Bool(true))));
        assert!(matches!(obj.get("nil"), Some(Document::Null)));
        assert!(matches!(obj.get("arr"), Some(Document::Array(a)) if a.len() == 2));
    }

    #[test]
    fn test_format_bedrock_error_adds_retention_hint() {
        let msg = "data retention mode 'default' is not available for this model";
        let out = format_bedrock_error(msg);
        assert!(out.contains("data-retention.html"));
    }

    #[test]
    fn test_format_bedrock_error_passthrough() {
        assert_eq!(format_bedrock_error("some other error"), "some other error");
    }
}
