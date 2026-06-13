# rs-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Rust port of [@earendil-works/pi-ai](https://www.npmjs.com/package/@earendil-works/pi-ai) — unified LLM API with automatic model discovery, streaming, tool calling, and multi-provider support.

> **⚠️ Early scaffold.** This crate is in initial development. The type system, event protocol, and registry are in place; provider implementations are being ported.

## Architecture

```
rs-ai/
├── src/
│   ├── lib.rs              # Crate root + re-exports
│   ├── types.rs            # Core types (Message, Context, Model, etc.)
│   ├── events.rs           # Stream event enum
│   ├── registry.rs         # Provider/model registry + stream/complete API
│   ├── env.rs              # Environment-based API key resolution
│   ├── compat.rs           # OpenAI-compatible provider detection
│   ├── models_generated.rs # Generated model registry (placeholder)
│   ├── provider/           # Provider implementations
│   │   └── openai.rs       # OpenAI Completions (placeholder)
│   ├── transports/
│   │   └── sse.rs          # SSE parser with sticky-field spec compliance
│   └── images/
│       ├── mod.rs
│       └── types.rs        # Image generation types
├── scripts/                # Code generator (model registry)
├── Cargo.toml
└── README.md
```

## Status

| Component | Status |
|---|---|
| Core types | ✅ Implemented |
| Event types | ✅ Implemented |
| Registry (model + provider) | ✅ Implemented |
| Env key resolution | ✅ Implemented |
| Compat detection | ✅ Implemented |
| SSE transport | ✅ Implemented + tested |
| Image types | ✅ Implemented |
| Message transform | ✅ Implemented + tested |
| Simple options / thinking | ✅ Implemented |
| Retry logic | ✅ Implemented + tested |
| Logger | ✅ Implemented |
| Diagnostics | ✅ Implemented |
| Azure normalization | ✅ Implemented + tested |
| Session resources | ✅ Implemented + tested |
| Prompt cache helpers | ✅ Implemented + tested |
| Input validation | ✅ Implemented + tested |
| Context overflow | ✅ Implemented + tested |
| OpenRouter image gen | ✅ Implemented |
| OpenAI provider | ✅ Streaming implemented |
| OpenAI Responses | ✅ Streaming implemented |
| Anthropic provider | ✅ Streaming implemented |
| Google Gemini | ✅ Streaming implemented |
| Mistral | ✅ Streaming implemented |
| Faux (test double) | ✅ Implemented + tested |
| Partial JSON parser | ✅ Implemented + tested |
| Harness helpers | ✅ Implemented + tested |
| Bedrock | ✅ Implemented (AWS SDK) |
| Codex (WebSocket + SSE) | ✅ Implemented |
| Gemini CLI | ✅ Implemented |
| OAuth flows | ✅ Framework + PKCE |

## Known limitations

Tracks `@earendil-works/pi-ai` `0.79.2`. Known divergences from upstream:

- **Google Vertex AI auth**: response decoding matches Gemini, but production Vertex
  auth (GCP Application Default Credentials / service-account token exchange and the
  project/location-scoped endpoint) is not implemented — it would require a GCP auth
  dependency. Vertex models fall back to the shared Gemini request path.
- **Provider SDK retries**: upstream relies on vendor SDK retry behavior. This port
  honors `StreamOptions` retry fields (`max_retries`, `max_retry_delay_ms`,
  `retry_config`) via `retry::do_with_retry` across the HTTP providers; Bedrock uses
  the AWS SDK's own retry. There is no implicit default retry when no options are set.

## Credits

Rust port of [**@earendil-works/pi-ai**](https://www.npmjs.com/package/@earendil-works/pi-ai), originally by [Mario Zechner](https://mariozechner.at).
