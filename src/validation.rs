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
