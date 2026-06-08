//! Input validation helpers.

use crate::types::{Context, Tool};

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validate a context before sending to a provider.
pub fn validate_context(ctx: &Context) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    if ctx.messages.is_empty() {
        errors.push(ValidationError {
            field: "messages".into(),
            message: "at least one message is required".into(),
        });
    }

    // Check tool definitions have valid JSON Schema parameters
    for (i, tool) in ctx.tools.iter().enumerate() {
        if let Err(e) = validate_tool(tool) {
            errors.push(ValidationError {
                field: format!("tools[{}]", i),
                message: e,
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_tool(tool: &Tool) -> Result<(), String> {
    if tool.name.is_empty() {
        return Err("tool name is required".into());
    }
    if tool.description.is_empty() {
        return Err("tool description is required".into());
    }
    // Parameters must be a valid JSON object (schema)
    if !tool.parameters.is_object() && !tool.parameters.is_null() {
        return Err("parameters must be a JSON object (schema)".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Context, Tool, user_message};
    use serde_json::json;

    #[test]
    fn test_valid_context() {
        let ctx = Context {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![user_message("hi")],
            tools: vec![Tool {
                name: "search".into(),
                description: "Search the web".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
        };
        assert!(validate_context(&ctx).is_ok());
    }

    #[test]
    fn test_empty_messages() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![],
            tools: vec![],
        };
        let errs = validate_context(&ctx).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].message.contains("at least one message"));
    }

    #[test]
    fn test_invalid_tool() {
        let ctx = Context {
            system_prompt: None,
            messages: vec![user_message("hi")],
            tools: vec![Tool {
                name: "".into(),
                description: "desc".into(),
                parameters: json!({}),
            }],
        };
        let errs = validate_context(&ctx).unwrap_err();
        assert!(errs[0].message.contains("name"));
    }
}

/// Validate a tool call's arguments against a schema (basic type check).
pub fn validate_tool_arguments(tool: &crate::types::Tool, args: &serde_json::Value) -> Result<(), String> {
    if tool.parameters.is_object() && !args.is_object() {
        return Err("arguments must be an object".into());
    }
    Ok(())
}

/// Validate a tool call (name exists in context tools, args valid).
pub fn validate_tool_call(ctx: &crate::types::Context, name: &str, args: &serde_json::Value) -> Result<(), String> {
    let tool = ctx.tools.iter().find(|t| t.name == name);
    match tool {
        Some(t) => validate_tool_arguments(t, args),
        None => Err(format!("tool '{}' not found in context", name)),
    }
}

/// Tool call limit configuration.
#[derive(Debug, Clone)]
pub struct ToolCallLimitConfig {
    pub max_parallel_calls: usize,
}

impl Default for ToolCallLimitConfig {
    fn default() -> Self {
        Self { max_parallel_calls: 128 }
    }
}

/// Default tool call limit config.
pub fn default_tool_call_limit_config() -> ToolCallLimitConfig {
    ToolCallLimitConfig::default()
}

/// Apply tool call limit (truncate excess calls).
pub fn apply_tool_call_limit(calls: &[serde_json::Value], config: &ToolCallLimitConfig) -> Vec<serde_json::Value> {
    if calls.len() <= config.max_parallel_calls {
        calls.to_vec()
    } else {
        calls[..config.max_parallel_calls].to_vec()
    }
}
