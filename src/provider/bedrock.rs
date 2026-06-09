//! Amazon Bedrock ConverseStream provider.

use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContent, ConversationRole, Message as BedrockMessage,
    SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;

use crate::events::Event;
use crate::types::*;

/// Start a Bedrock ConverseStream.
pub fn stream_bedrock<'a>(
    model: &'a Model,
    context: &'a Context,
    _opts: &'a StreamOptions,
) -> std::pin::Pin<Box<dyn futures::Stream<Item = Event> + Send + 'a>> {
    Box::pin(async_stream::stream! {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let client = BedrockClient::new(&config);

        let mut messages = Vec::new();
        for msg in &context.messages {
            let role = match msg.role {
                Role::User | Role::ToolResult => ConversationRole::User,
                Role::Assistant => ConversationRole::Assistant,
                _ => continue,
            };
            let content: Vec<BedrockContent> = msg.content.iter().filter_map(|b| match b {
                ContentBlock::Text { text, .. } => {
                    Some(BedrockContent::Text(text.clone()))
                }
                _ => None,
            }).collect();
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

        let result = req.send().await;

        let mut output = match result {
            Ok(o) => o,
            Err(e) => {
                yield Event::Error {
                    reason: StopReason::Error,
                    error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(e.to_string())),
                    message: None,
                };
                return;
            }
        };

        let mut partial = Message {
            role: Role::Assistant,
            content: Vec::new(),
            timestamp: 0,
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

        let mut recv = output.stream;
        loop {
            match recv.recv().await {
                Ok(Some(event)) => {
                    use aws_sdk_bedrockruntime::types::ConverseStreamOutput;
                    match event {
                        ConverseStreamOutput::ContentBlockStart(_) => {
                            if !text_started {
                                text_started = true;
                                yield Event::TextStart;
                            }
                        }
                        ConverseStreamOutput::ContentBlockDelta(delta) => {
                            if let Some(d) = delta.delta() {
                                if let aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(t) = d {
                                    current_text.push_str(t);
                                    yield Event::TextDelta { delta: t.to_string() };
                                }
                            }
                        }
                        ConverseStreamOutput::ContentBlockStop(_) => {
                            if text_started {
                                text_started = false;
                                yield Event::TextEnd;
                            }
                        }
                        ConverseStreamOutput::MessageStop(stop) => {
                            let reason = stop.stop_reason();
                            partial.stop_reason = Some(match reason {
                                aws_sdk_bedrockruntime::types::StopReason::EndTurn => StopReason::Stop,
                                aws_sdk_bedrockruntime::types::StopReason::MaxTokens => StopReason::Length,
                                aws_sdk_bedrockruntime::types::StopReason::ToolUse => StopReason::ToolUse,
                                _ => StopReason::Stop,
                            });
                        }
                        ConverseStreamOutput::Metadata(meta) => {
                            if let Some(u) = meta.usage() {
                                partial.usage = Some(Usage {
                                    input: u.input_tokens() as u32,
                                    output: u.output_tokens() as u32,
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
                        error: Arc::from(Box::<dyn std::error::Error + Send + Sync>::from(e.to_string())),
                        message: Some(partial.clone()),
                    };
                    return;
                }
            }
        }

        if !current_text.is_empty() {
            partial.content.push(ContentBlock::Text { text: current_text, text_signature: None });
        }
        let reason = partial.stop_reason.clone().unwrap_or(StopReason::Stop);
        yield Event::Done { reason, message: partial };
    })
}
