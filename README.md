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
| Model generator | ✅ 968 models / 35 providers |
| OpenAI provider | ✅ Streaming implemented |
| Other providers | 🔲 Not started |

## Credits

Rust port of [**@earendil-works/pi-ai**](https://www.npmjs.com/package/@earendil-works/pi-ai), originally by [Mario Zechner](https://mariozechner.at).
