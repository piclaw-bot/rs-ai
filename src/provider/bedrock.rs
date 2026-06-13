//! Amazon Bedrock ConverseStream provider.

use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, AutoToolChoice, CachePointBlock, CachePointType, CacheTtl,
    ContentBlock as BedrockContent, ConversationRole, ImageBlock, ImageFormat, ImageSource,
    Message as BedrockMessage, ReasoningContentBlock, ReasoningTextBlock, SpecificToolChoice,
    SystemContentBlock, Tool, ToolChoice, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolResultStatus, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_types::{Document, Number};

use crate::events::Event;
use crate::types::*;

const EMPTY_TEXT_PLACEHOLDER: &str = "<empty>";

/// Whether the Bedrock model supports prompt caching (mirrors supportsPromptCaching).
fn supports_bedrock_prompt_caching(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let name = model.name.to_lowercase();
    let has_claude = id.contains("claude") || name.contains("claude");
    if !has_claude {
        return std::env::var("AWS_BEDROCK_FORCE_CACHE").ok().as_deref() == Some("1");
    }
    let m = |needle: &str| id.contains(needle) || name.contains(needle);
    m("-4-") || m("claude-3-7-sonnet") || m("claude-3-5-haiku")
}

/// Build a Bedrock cache-point block with an optional 1h TTL for long retention.
fn bedrock_cache_point(long: bool) -> CachePointBlock {
    let mut b = CachePointBlock::builder().r#type(CachePointType::Default);
    if long {
        b = b.ttl(CacheTtl::OneHour);
    }
    b.build().unwrap()
}

/// True for Anthropic Claude models on Bedrock (id or name), which support the
/// reasoningContent signature field (mirrors isAnthropicClaudeModel).
fn is_anthropic_claude_model(model: &Model) -> bool {
    let id = model.id.to_lowercase();
    let name = model.name.to_lowercase();
    id.contains("anthropic.claude")
        || id.contains("anthropic/claude")
        || name.contains("anthropic.claude")
        || name.contains("anthropic/claude")
        || name.contains("claude")
}

/// Sanitize a tool-call id for Bedrock (alnum/_/- only, max 64 chars).
fn normalize_bedrock_tool_call_id(id: &str) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if sanitized.len() > 64 { sanitized[..64].to_string() } else { sanitized }
}

/// Build a non-blank text content block, or None when blank (mirrors createNonBlankTextBlock).
fn non_blank_text(text: &str) -> Option<BedrockContent> {
    if text.trim().is_empty() {
        None
    } else {
        Some(BedrockContent::Text(text.to_string()))
    }
}

/// Build a Bedrock image block from a base64 data string.
fn bedrock_image_block(mime_type: &str, data: &str) -> Option<ImageBlock> {
    use base64::Engine;
    let format = match mime_type {
        "image/jpeg" | "image/jpg" => ImageFormat::Jpeg,
        "image/png" => ImageFormat::Png,
        "image/gif" => ImageFormat::Gif,
        "image/webp" => ImageFormat::Webp,
        _ => return None,
    };
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).ok()?;
    ImageBlock::builder()
        .format(format)
        .source(ImageSource::Bytes(aws_smithy_types::Blob::new(bytes)))
        .build()
        .ok()
}

/// Convert tool-result content into Bedrock tool-result content blocks, mirroring
/// convertToolResultContent (images + non-blank text, with an empty placeholder fallback).
fn convert_tool_result_content(content: &[ContentBlock]) -> Vec<ToolResultContentBlock> {
    let mut result: Vec<ToolResultContentBlock> = Vec::new();
    for c in content {
        match c {
            ContentBlock::Image { data, mime_type } => {
                if let Some(img) = bedrock_image_block(mime_type, data) {
                    result.push(ToolResultContentBlock::Image(img));
                }
            }
            ContentBlock::Text { text, .. } if !text.trim().is_empty() => {
                result.push(ToolResultContentBlock::Text(text.clone()));
            }
            _ => {}
        }
    }
    if result.is_empty() {
        result.push(ToolResultContentBlock::Text(EMPTY_TEXT_PLACEHOLDER.to_string()));
    }
    result
}

/// Build the Bedrock `additionalModelRequestFields` thinking config for Anthropic
/// Claude models (mirrors buildAdditionalModelRequestFields).
fn bedrock_thinking_fields(model: &Model, opts: &StreamOptions) -> Option<serde_json::Value> {
    if !model.reasoning || !is_anthropic_claude_model(model) {
        return None;
    }
    let level = opts.reasoning.as_ref()?;
    let key = format!("{level:?}").to_lowercase();
    if model.compat.force_adaptive_thinking == Some(true) {
        // Adaptive-thinking models: effort-based config, no interleaved beta.
        let default_effort = match key.as_str() {
            "minimal" | "low" => "low",
            "medium" => "medium",
            _ => "high",
        };
        let effort = model.thinking_level_map.as_ref()
            .and_then(|m| m.get(&key)).and_then(|v| v.clone())
            .unwrap_or_else(|| default_effort.to_string());
        Some(serde_json::json!({
            "thinking": { "type": "adaptive", "display": "summarized" },
            "output_config": { "effort": effort },
        }))
    } else {
        let default_budget = match key.as_str() {
            "minimal" => 1024, "low" => 2048, "medium" => 8192, _ => 16384,
        };
        let budget = opts.thinking_budgets.as_ref().and_then(|b| match key.as_str() {
            "minimal" => b.minimal, "low" => b.low, "medium" => b.medium, _ => b.high,
        }).unwrap_or(default_budget);
        Some(serde_json::json!({
            "thinking": { "type": "enabled", "budget_tokens": budget, "display": "summarized" },
            "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        }))
    }
}

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

        let supports_signature = is_anthropic_claude_model(model);
        let transformed = crate::transform::transform_messages(&context.messages, model);
        let mut messages = Vec::new();
        let mut i = 0;
        while i < transformed.len() {
            let msg = &transformed[i];
            match msg.role {
                Role::User => {
                    let mut content: Vec<BedrockContent> = Vec::new();
                    for b in &msg.content {
                        match b {
                            ContentBlock::Text { text, .. } => {
                                if let Some(tb) = non_blank_text(text) {
                                    content.push(tb);
                                }
                            }
                            ContentBlock::Image { data, mime_type } => {
                                if let Some(img) = bedrock_image_block(mime_type, data) {
                                    content.push(BedrockContent::Image(img));
                                }
                            }
                            _ => {}
                        }
                    }
                    if content.is_empty() {
                        content.push(BedrockContent::Text(EMPTY_TEXT_PLACEHOLDER.to_string()));
                    }
                    messages.push(BedrockMessage::builder().role(ConversationRole::User).set_content(Some(content)).build().unwrap());
                    i += 1;
                }
                Role::Assistant => {
                    if msg.content.is_empty() { i += 1; continue; }
                    let mut content: Vec<BedrockContent> = Vec::new();
                    for b in &msg.content {
                        match b {
                            ContentBlock::Text { text, .. } => {
                                if let Some(tb) = non_blank_text(text) {
                                    content.push(tb);
                                }
                            }
                            ContentBlock::ToolCall { id, name, arguments, .. } => {
                                let args_value = serde_json::to_value(arguments).unwrap_or_else(|_| serde_json::json!({}));
                                if let Ok(tub) = ToolUseBlock::builder()
                                    .tool_use_id(normalize_bedrock_tool_call_id(id))
                                    .name(name.clone())
                                    .input(json_to_document(&args_value))
                                    .build()
                                {
                                    content.push(BedrockContent::ToolUse(tub));
                                }
                            }
                            ContentBlock::Thinking { thinking, thinking_signature, redacted } if !redacted && !thinking.trim().is_empty() => {
                                // Only Anthropic Claude models accept the reasoning signature.
                                // For Claude with a missing signature, fall back to plain text
                                // (Bedrock rejects a replayed reasoning block without a signature).
                                if supports_signature {
                                    match thinking_signature.as_ref().filter(|s| !s.trim().is_empty()) {
                                        Some(sig) => {
                                            if let Ok(rt) = ReasoningTextBlock::builder().text(thinking.clone()).signature(sig.clone()).build() {
                                                content.push(BedrockContent::ReasoningContent(ReasoningContentBlock::ReasoningText(rt)));
                                            }
                                        }
                                        None => content.push(BedrockContent::Text(thinking.clone())),
                                    }
                                } else if let Ok(rt) = ReasoningTextBlock::builder().text(thinking.clone()).build() {
                                    content.push(BedrockContent::ReasoningContent(ReasoningContentBlock::ReasoningText(rt)));
                                }
                            }
                            _ => {}
                        }
                    }
                    if content.is_empty() { i += 1; continue; }
                    messages.push(BedrockMessage::builder().role(ConversationRole::Assistant).set_content(Some(content)).build().unwrap());
                    i += 1;
                }
                Role::ToolResult => {
                    // Merge consecutive tool results into a single user message.
                    let mut content: Vec<BedrockContent> = Vec::new();
                    while i < transformed.len() && transformed[i].role == Role::ToolResult {
                        let tr = &transformed[i];
                        let status = if tr.is_error { ToolResultStatus::Error } else { ToolResultStatus::Success };
                        if let Ok(trb) = ToolResultBlock::builder()
                            .tool_use_id(normalize_bedrock_tool_call_id(tr.tool_call_id.as_deref().unwrap_or_default()))
                            .set_content(Some(convert_tool_result_content(&tr.content)))
                            .status(status)
                            .build()
                        {
                            content.push(BedrockContent::ToolResult(trb));
                        }
                        i += 1;
                    }
                    messages.push(BedrockMessage::builder().role(ConversationRole::User).set_content(Some(content)).build().unwrap());
                }
            }
        }

        // Prompt caching: add cache points to the last user message and the system prompt
        // for supported Claude models (mirrors the cachePoint logic).
        let cache_long = matches!(opts.cache_retention, Some(CacheRetention::Long));
        let cache_enabled = !matches!(opts.cache_retention, Some(CacheRetention::None) | None)
            && supports_bedrock_prompt_caching(model);
        if cache_enabled
            && let Some(last) = messages.pop() {
            if last.role() == &ConversationRole::User {
                let mut content = last.content().to_vec();
                content.push(BedrockContent::CachePoint(bedrock_cache_point(cache_long)));
                messages.push(BedrockMessage::builder().role(ConversationRole::User).set_content(Some(content)).build().unwrap());
            } else {
                messages.push(last);
            }
        }

        let mut req = client
            .converse_stream()
            .model_id(&model.id)
            .set_messages(Some(messages));

        if let Some(ref prompt) = context.system_prompt {
            req = req.system(SystemContentBlock::Text(prompt.clone()));
            if cache_enabled {
                req = req.system(SystemContentBlock::CachePoint(bedrock_cache_point(cache_long)));
            }
        }

        // Inference config: max output tokens (defaults to the model cap for Claude) and
        // temperature (mirrors inferenceConfig).
        let inference_max_tokens = opts.max_tokens.or_else(|| {
            if is_anthropic_claude_model(model) && model.max_tokens > 0 { Some(model.max_tokens) } else { None }
        });
        if inference_max_tokens.is_some() || opts.temperature.is_some() {
            let mut ic = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();
            if let Some(mt) = inference_max_tokens {
                ic = ic.max_tokens(mt as i32);
            }
            if let Some(temp) = opts.temperature {
                ic = ic.temperature(temp as f32);
            }
            req = req.inference_config(ic.build());
        }

        // Tool config: skip entirely when toolChoice is "none" (mirrors convertToolConfig).
        let tool_choice_none = opts.tool_choice.as_ref().and_then(|v| v.as_str()) == Some("none");
        if !context.tools.is_empty() && !tool_choice_none {
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
            let mut tc_builder = ToolConfiguration::builder().set_tools(Some(tool_list));
            // Map tool choice: auto/any/{type:tool,name}.
            if let Some(choice) = opts.tool_choice.as_ref() {
                if let Some(s) = choice.as_str() {
                    match s {
                        "auto" => tc_builder = tc_builder.tool_choice(ToolChoice::Auto(AutoToolChoice::builder().build())),
                        "any" | "required" => tc_builder = tc_builder.tool_choice(ToolChoice::Any(AnyToolChoice::builder().build())),
                        _ => {}
                    }
                } else if choice.get("type").and_then(|v| v.as_str()) == Some("tool")
                    && let Some(name) = choice.get("name").and_then(|v| v.as_str())
                    && let Ok(spec) = SpecificToolChoice::builder().name(name).build() {
                    tc_builder = tc_builder.tool_choice(ToolChoice::Tool(spec));
                }
            }
            if let Ok(tc) = tc_builder.build() {
                req = req.tool_config(tc);
            }
        }

        // Enable thinking for Anthropic Claude models on Bedrock (additionalModelRequestFields).
        if let Some(fields) = bedrock_thinking_fields(model, opts) {
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

    #[test]
    fn test_is_anthropic_claude_model() {
        use super::is_anthropic_claude_model;
        use crate::types::{Model, ModelCost};
        let mk = |id: &str, name: &str| Model {
            id: id.into(), name: name.into(), api: "bedrock-converse-stream".into(),
            provider: "bedrock".into(), base_url: String::new(), reasoning: true,
            thinking_level_map: None, input: vec!["text".into()], cost: ModelCost::default(),
            context_window: 0, max_tokens: 0, headers: None, api_key: None, compat: Default::default(),
        };
        assert!(is_anthropic_claude_model(&mk("anthropic.claude-sonnet-4", "")));
        assert!(is_anthropic_claude_model(&mk("some-profile", "Anthropic Claude Sonnet")));
        assert!(!is_anthropic_claude_model(&mk("meta.llama3", "Llama 3")));
    }

    #[test]
    fn test_convert_tool_result_content_empty_and_text() {
        use super::{convert_tool_result_content, EMPTY_TEXT_PLACEHOLDER};
        use crate::types::ContentBlock;
        use aws_sdk_bedrockruntime::types::ToolResultContentBlock;
        // Empty content -> placeholder.
        let out = convert_tool_result_content(&[]);
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], ToolResultContentBlock::Text(t) if t == EMPTY_TEXT_PLACEHOLDER));
        // Blank text is skipped, real text kept.
        let out = convert_tool_result_content(&[
            ContentBlock::Text { text: "   ".into(), text_signature: None },
            ContentBlock::Text { text: "done".into(), text_signature: None },
        ]);
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], ToolResultContentBlock::Text(t) if t == "done"));
    }

    #[test]
    fn test_normalize_bedrock_tool_call_id() {
        use super::normalize_bedrock_tool_call_id;
        assert_eq!(normalize_bedrock_tool_call_id("call:1|x"), "call_1_x");
        assert_eq!(normalize_bedrock_tool_call_id("abc-123_OK"), "abc-123_OK");
        assert_eq!(normalize_bedrock_tool_call_id(&"a".repeat(80)).len(), 64);
    }

    #[test]
    fn test_bedrock_thinking_fields() {
        use super::bedrock_thinking_fields;
        use crate::types::{Model, ModelCost, StreamOptions, ThinkingLevel};
        let mk = |id: &str, adaptive: bool| {
            let mut m = Model {
                id: id.into(), name: String::new(), api: "bedrock-converse-stream".into(),
                provider: "bedrock".into(), base_url: String::new(), reasoning: true,
                thinking_level_map: None, input: vec!["text".into()], cost: ModelCost::default(),
                context_window: 0, max_tokens: 0, headers: None, api_key: None, compat: Default::default(),
            };
            if adaptive { m.compat.force_adaptive_thinking = Some(true); }
            m
        };
        // Budget-based (non-adaptive) Claude: enabled + interleaved beta.
        let opts = StreamOptions { reasoning: Some(ThinkingLevel::High), ..Default::default() };
        let f = bedrock_thinking_fields(&mk("anthropic.claude-3", false), &opts).unwrap();
        assert_eq!(f["thinking"]["type"], "enabled");
        assert_eq!(f["thinking"]["budget_tokens"], 16384);
        assert!(f["anthropic_beta"].is_array());
        // Adaptive Claude: adaptive + output_config, no interleaved beta.
        let f = bedrock_thinking_fields(&mk("anthropic.claude-opus-4-6", true), &opts).unwrap();
        assert_eq!(f["thinking"]["type"], "adaptive");
        assert_eq!(f["output_config"]["effort"], "high");
        assert!(f.get("anthropic_beta").is_none());
        // Non-Claude model: no thinking fields.
        let f = bedrock_thinking_fields(&mk("meta.llama3", false), &opts);
        assert!(f.is_none());
        // No reasoning requested: none.
        let f = bedrock_thinking_fields(&mk("anthropic.claude-3", false), &StreamOptions::default());
        assert!(f.is_none());
    }

    #[test]
    fn test_supports_bedrock_prompt_caching() {
        use super::supports_bedrock_prompt_caching;
        use crate::types::{Model, ModelCost};
        let mk = |id: &str, name: &str| Model {
            id: id.into(), name: name.into(), api: "bedrock-converse-stream".into(),
            provider: "bedrock".into(), base_url: String::new(), reasoning: false,
            thinking_level_map: None, input: vec!["text".into()], cost: ModelCost::default(),
            context_window: 0, max_tokens: 0, headers: None, api_key: None, compat: Default::default(),
        };
        assert!(supports_bedrock_prompt_caching(&mk("anthropic.claude-sonnet-4-5", "")));
        assert!(supports_bedrock_prompt_caching(&mk("anthropic.claude-3-7-sonnet", "")));
        assert!(supports_bedrock_prompt_caching(&mk("anthropic.claude-3-5-haiku", "")));
        assert!(!supports_bedrock_prompt_caching(&mk("anthropic.claude-3-sonnet", "")));
        assert!(!supports_bedrock_prompt_caching(&mk("meta.llama3", "Llama")));
    }
}
