#!/usr/bin/env python3
"""Generate src/models_generated.rs from upstream models JSON.

Usage:
    # First dump models to JSON via bun:
    bun --eval "import { MODELS } from 'path/to/models.generated.js'; process.stdout.write(JSON.stringify(MODELS));" > /tmp/models.json
    # Then generate:
    python3 scripts/generate_models.py /tmp/models.json
"""

import json
import sys
import datetime
from pathlib import Path

def rust_string(s):
    return json.dumps(s)

def gen_model(m) -> str:
    lines = []
    lines.append("        Model {")
    lines.append(f'            id: {rust_string(m["id"])}.into(),')
    lines.append(f'            name: {rust_string(m["name"])}.into(),')
    lines.append(f'            api: {rust_string(m["api"])}.into(),')
    lines.append(f'            provider: {rust_string(m["provider"])}.into(),')
    lines.append(f'            base_url: {rust_string(m.get("baseUrl", ""))}.into(),')
    lines.append(f'            reasoning: {str(m.get("reasoning", False)).lower()},')
    
    tlm = m.get("thinkingLevelMap")
    if tlm:
        entries = []
        for k, v in tlm.items():
            if v is None:
                entries.append(f'                ({rust_string(k)}.into(), None)')
            else:
                entries.append(f'                ({rust_string(k)}.into(), Some({rust_string(v)}.into()))')
        lines.append("            thinking_level_map: Some(HashMap::from([")
        lines.append(",\n".join(entries))
        lines.append("            ])),")
    else:
        lines.append("            thinking_level_map: None,")
    
    inputs = m.get("input", [])
    input_str = ", ".join(f'{rust_string(i)}.into()' for i in inputs)
    lines.append(f"            input: vec![{input_str}],")
    
    cost = m.get("cost", {})
    ci = cost.get("input", 0)
    co = cost.get("output", 0)
    cr = cost.get("cacheRead", 0)
    cw = cost.get("cacheWrite", 0)
    lines.append(f"            cost: ModelCost {{ input: {ci}_f64, output: {co}_f64, cache_read: {cr}_f64, cache_write: {cw}_f64 }},")
    lines.append(f"            context_window: {m.get('contextWindow', 0)},")
    lines.append(f"            max_tokens: {m.get('maxTokens', 0)},")
    
    headers = m.get("headers")
    if headers:
        entries = ", ".join(f'({rust_string(k)}.into(), {rust_string(v)}.into())' for k, v in headers.items())
        lines.append(f"            headers: Some(HashMap::from([{entries}])),")
    else:
        lines.append("            headers: None,")
    
    lines.append("            api_key: None,")
    compat = m.get("compat") or {}
    compat_fields = {
        "forceAdaptiveThinking": ("force_adaptive_thinking", "bool"),
        "maxTokensField": ("max_tokens_field", "str"),
        "requiresReasoningContentOnAssistantMessages": ("requires_reasoning_content_on_assistant_messages", "bool"),
        "sendSessionAffinityHeaders": ("send_session_affinity_headers", "bool"),
        "supportsCacheControlOnTools": ("supports_cache_control_on_tools", "bool"),
        "supportsDeveloperRole": ("supports_developer_role", "bool"),
        "supportsEagerToolInputStreaming": ("supports_eager_tool_input_streaming", "bool"),
        "supportsLongCacheRetention": ("supports_long_cache_retention", "bool"),
        "supportsReasoningEffort": ("supports_reasoning_effort", "bool"),
        "supportsStore": ("supports_store", "bool"),
        "supportsStrictMode": ("supports_strict_mode", "bool"),
        "supportsTemperature": ("supports_temperature", "bool"),
        "thinkingFormat": ("thinking_format", "str"),
        "zaiToolStream": ("zai_tool_stream", "bool"),
    }
    compat_lines = []
    for js_key, (rs_key, kind) in compat_fields.items():
        if js_key in compat and compat[js_key] is not None:
            v = compat[js_key]
            if kind == "bool":
                compat_lines.append(f"                {rs_key}: Some({str(v).lower()}),")
            else:
                compat_lines.append(f"                {rs_key}: Some({rust_string(v)}.into()),")
    if compat_lines:
        lines.append("            compat: ModelCompat {")
        lines.extend(compat_lines)
        lines.append("                ..Default::default()")
        lines.append("            },")
    else:
        lines.append("            compat: ModelCompat::default(),")
    lines.append("        }")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/generate_models.py /tmp/models.json", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    models = json.loads(input_path.read_text())
    
    all_models = []
    for provider in sorted(models.keys()):
        for model_id in sorted(models[provider].keys()):
            all_models.append(models[provider][model_id])
    
    total = len(all_models)
    providers = len(models)
    now = datetime.datetime.utcnow().isoformat()
    
    out = []
    out.append(f"//! Auto-generated model registry from @earendil-works/pi-ai. DO NOT EDIT.")
    out.append(f"//!")
    out.append(f"//! Source: models.generated.js ({total} models, {providers} providers)")
    out.append(f"//! Generated: {now}Z")
    out.append("")
    out.append("#![allow(clippy::approx_constant)]")
    out.append("")
    out.append("use std::collections::HashMap;")
    out.append("use crate::types::{Model, ModelCost, ModelCompat};")
    out.append("")
    out.append("/// Returns all built-in models from the upstream pi-ai registry.")
    out.append("pub fn builtin_models() -> Vec<Model> {")
    out.append("    vec![")
    
    for m in all_models:
        out.append(gen_model(m) + ",")
    
    out.append("    ]")
    out.append("}")
    
    output_path = Path(__file__).parent.parent / "src" / "models_generated.rs"
    output_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {output_path} ({total} models, {providers} providers, {len(out)} lines)")

if __name__ == "__main__":
    main()
