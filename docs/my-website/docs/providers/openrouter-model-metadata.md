<!-- BEGIN openrouter-capability-table (generated) -->
<!--
  GENERATED FILE -- DO NOT EDIT MANUALLY
  Generator: scripts/generate_openrouter_capability_table.py
  Source: model_prices_and_context_window.json
  Source SHA256: 15db84df298415c1008b2d112c4efc2bf69390bb70b65c526e939b7b8fab8d6c
  Source snapshot date: 2026-07-20
  OpenRouter model count: 131
-->

# OpenRouter Model Metadata

> **LiteLLM metadata snapshot** -- not the live OpenRouter model catalog.
> Generated from `model_prices_and_context_window.json` by
> `scripts/generate_openrouter_capability_table.py`.
> Regenerate: `python scripts/generate_openrouter_capability_table.py`

- **Source snapshot date:** 2026-07-20
- **Source SHA256:** `15db84df298415c1008b2d112c4efc2bf69390bb70b65c526e939b7b8fab8d6c`
- **Model count:** 131

## Capability Legend

| Symbol | Meaning |
|--------|---------|
| Y | Explicitly declared as supported |
| N | Explicitly declared as not supported |
| - | Not declared in metadata |

### Column Abbreviations

| Abbrev | Capability |
|--------|------------|
| FC | supports_function_calling |
| Vis | supports_vision |
| Reas | supports_reasoning |
| Cch | supports_prompt_caching |
| TC | supports_tool_choice |
| AI | supports_audio_input |
| AO | supports_audio_output |
| PDF | supports_pdf_input |
| VI | supports_video_input |
| CU | supports_computer_use |
| RS | supports_response_schema |
| SM | supports_system_messages |
| PF | supports_parallel_function_calling |
| NC | supports_native_cache_control |
| AP | supports_assistant_prefill |
| UC | supports_url_context |
| WS | supports_web_search |

## Models

| Model | Mode | Ctx | MaxOut | FC | Vis | Reas | Cch | TC | AI | AO | PDF | VI | CU | RS | SM | PF | NC | AP | UC | WS | InMod | OutMod | Endpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openrouter/anthropic/claude-3-haiku | chat | 200000 | - | Y | Y | - | - | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |
| openrouter/anthropic/claude-3.5-sonnet | chat | 200000 | 8192 | Y | Y | - | - | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-3.7-sonnet | chat | 200000 | 128000 | Y | Y | Y | - | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-haiku-4.5 | chat | 200000 | 200000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-opus-4 | chat | 200000 | 32000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-opus-4.1 | chat | 200000 | 32000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-opus-4.5 | chat | 200000 | 32000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-opus-4.6 | chat | 1000000 | 128000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-sonnet-4 | chat | 1000000 | 64000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-sonnet-4.5 | chat | 1000000 | 1000000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/anthropic/claude-sonnet-4.6 | chat | 1000000 | 128000 | Y | Y | Y | Y | Y | - | - | - | - | Y | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/baai/bge-base-en-v1.5 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/baai/bge-large-en-v1.5 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/baai/bge-m3 | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/bytedance/ui-tars-1.5-7b | chat | 131072 | 2048 | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/cohere/rerank-4-fast | rerank | 32768 | 32768 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/cohere/rerank-4-pro | rerank | 32768 | 32768 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/cohere/rerank-v3.5 | rerank | 4096 | 4096 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/deepseek/deepseek-chat | chat | 65536 | 8192 | - | - | - | Y | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/deepseek/deepseek-chat-v3-0324 | chat | 65536 | 8192 | - | - | - | Y | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/deepseek/deepseek-chat-v3.1 | chat | 163840 | 163840 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/deepseek/deepseek-r1 | chat | 65336 | 8192 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/deepseek/deepseek-r1-0528 | chat | 65336 | 8192 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/deepseek/deepseek-v3.2 | chat | 163840 | 163840 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/deepseek/deepseek-v3.2-exp | chat | 163840 | 163840 | Y | - | N | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/deepseek/deepseek-v4-flash:free | chat | 1000000 | 1000000 | Y | - | N | Y | Y | - | - | - | - | - | - | - | - | - | Y | - | - | - | - | - |
| openrouter/free | chat | 200000 | - | Y | Y | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/google/gemini-2.0-flash-001 | chat | 1048576 | 8192 | Y | Y | - | - | Y | - | Y | - | - | - | Y | Y | - | Y | - | - | - | - | - | - |
| openrouter/google/gemini-2.5-flash | chat | 1048576 | 8192 | Y | Y | - | - | Y | - | Y | - | - | - | Y | Y | - | Y | - | - | - | - | - | - |
| openrouter/google/gemini-2.5-pro | chat | 1048576 | 8192 | Y | Y | - | - | Y | - | Y | - | - | - | Y | Y | - | Y | - | - | - | - | - | - |
| openrouter/google/gemini-3-flash-preview | chat | 1048576 | 65535 | Y | Y | Y | Y | Y | - | N | Y | - | - | Y | Y | Y | Y | - | Y | Y | audio, image, text, video | text | /v1/batch, /v1/chat/completions, /v1/completions |
| openrouter/google/gemini-3-pro-preview | chat | 1048576 | 65535 | Y | Y | Y | Y | Y | Y | - | Y | Y | - | Y | Y | - | Y | - | - | Y | audio, image, text, video | text | /v1/batch, /v1/chat/completions, /v1/completions |
| openrouter/google/gemini-3.1-pro-preview | chat | 1048576 | 65536 | Y | Y | Y | Y | Y | Y | - | Y | - | - | Y | Y | - | Y | - | - | - | audio, image, text, video | text | - |
| openrouter/google/gemini-embedding-001 | embedding | 20000 | 20000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/google/gemini-embedding-2-preview | embedding | 8192 | 8192 | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/google/gemma-4-26b-a4b-it:free | chat | 262144 | - | Y | Y | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/google/gemma-4-31b-it:free | chat | 262144 | - | Y | Y | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/gpt-oss-120b:free | chat | 131072 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/gpt-oss-20b:free | chat | 131072 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/gryphe/mythomax-l2-13b | chat | 8192 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/inclusionai/ling-2.6-flash | chat | 262144 | - | Y | - | - | Y | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/inclusionai/ling-2.6-flash:free | chat | 262144 | - | Y | - | - | Y | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/intfloat/e5-base-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/intfloat/e5-large-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/intfloat/multilingual-e5-large | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mancer/weaver | chat | 8000 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/meta-llama/llama-3-70b-instruct | chat | 8192 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/meta-llama/llama-3.3-70b-instruct:free | chat | 65536 | - | Y | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/minimax/minimax-m2 | chat | 204800 | 204800 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |
| openrouter/minimax/minimax-m2.1 | chat | 204000 | 64000 | Y | Y | Y | N | Y | - | - | - | - | N | - | - | - | Y | - | - | - | - | - | - |
| openrouter/minimax/minimax-m2.5 | chat | 196608 | 65536 | Y | N | Y | Y | Y | - | - | - | - | N | - | - | - | Y | - | - | - | - | - | - |
| openrouter/minimax/minimax-m2.5:free | chat | 196608 | - | Y | - | Y | - | N | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/codestral-embed-2505 | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/devstral-2512 | chat | 262144 | 65536 | Y | N | - | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/ministral-14b-2512 | chat | 262144 | 262144 | Y | Y | - | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/ministral-3b-2512 | chat | 131072 | 131072 | Y | Y | - | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/ministral-8b-2512 | chat | 262144 | 262144 | Y | Y | - | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-7b-instruct | chat | 8192 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-embed-2312 | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-large | chat | 32000 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-large-2512 | chat | 262144 | 262144 | Y | Y | - | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-small-3.1-24b-instruct | chat | 32000 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mistral-small-3.2-24b-instruct | chat | 32000 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/mistralai/mixtral-8x22b-instruct | chat | 65536 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/moonshotai/kimi-k2.5 | chat | 262144 | 262144 | Y | Y | - | - | Y | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free | embedding | 131072 | 131072 | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/nvidia/nemotron-3-super-120b-a12b:free | chat | 262144 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-3.5-turbo | chat | 4095 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-3.5-turbo-16k | chat | 16383 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4 | chat | 8192 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4.1 | chat | 1047576 | 32768 | Y | Y | - | Y | Y | - | - | - | - | - | Y | Y | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4.1-mini | chat | 1047576 | 32768 | Y | Y | - | Y | Y | - | - | - | - | - | Y | Y | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4.1-nano | chat | 1047576 | 32768 | Y | Y | - | Y | Y | - | - | - | - | - | Y | Y | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4o | chat | 128000 | 4096 | Y | Y | - | - | Y | - | - | - | - | - | - | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-4o-2024-05-13 | chat | 128000 | 4096 | Y | Y | - | - | Y | - | - | - | - | - | - | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-5 | chat | 272000 | 128000 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5-chat | chat | 128000 | 16384 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5-codex | chat | 272000 | 128000 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5-mini | chat | 272000 | 128000 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5-nano | chat | 272000 | 128000 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5.1-codex-max | chat | 400000 | 128000 | Y | Y | Y | Y | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5.2 | chat | 272000 | 128000 | Y | Y | Y | Y | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-5.2-chat | chat | 128000 | 16384 | Y | Y | - | Y | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-5.2-codex | chat | 272000 | 128000 | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | image, text | text | - |
| openrouter/openai/gpt-5.2-pro | chat | 272000 | 128000 | Y | Y | Y | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-oss-120b | chat | 131072 | 32768 | Y | - | Y | - | Y | - | - | - | - | - | Y | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-oss-120b:free | chat | 131072 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/openai/gpt-oss-20b | chat | 131072 | 32768 | Y | - | Y | - | Y | - | - | - | - | - | Y | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/gpt-oss-20b:free | chat | 131072 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/openai/o1 | chat | 200000 | 100000 | Y | Y | - | Y | Y | - | - | - | - | - | Y | Y | Y | - | - | - | - | - | - | - |
| openrouter/openai/o3-mini | chat | 128000 | 65536 | Y | N | Y | - | Y | - | - | - | - | - | - | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/o3-mini-high | chat | 128000 | 65536 | Y | N | Y | - | Y | - | - | - | - | - | - | - | Y | - | - | - | - | - | - | - |
| openrouter/openai/text-embedding-3-large | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/text-embedding-3-small | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openai/text-embedding-ada-002 | embedding | 8192 | 8192 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openrouter/auto | chat | 2000000 | - | Y | Y | Y | - | Y | Y | - | - | Y | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/openrouter/bodybuilder | chat | 128000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/openrouter/free | chat | 200000 | - | Y | Y | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/perplexity/pplx-embed-v1-0.6b | embedding | 32000 | 32000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/perplexity/pplx-embed-v1-4b | embedding | 32000 | 32000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen-2.5-coder-32b-instruct | chat | 33792 | 33792 | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen-vl-plus | chat | 8192 | 2048 | - | Y | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-235b-a22b-2507 | chat | 262144 | 262144 | Y | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-235b-a22b-thinking-2507 | chat | 262144 | 262144 | Y | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-coder | chat | 262100 | 262100 | Y | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-coder-plus | chat | 997952 | 65536 | Y | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-coder:free | chat | 262000 | - | Y | - | Y | - | Y | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-embedding-4b | embedding | 32768 | 32768 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3-embedding-8b | embedding | 32000 | 32000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-122b-a10b | chat | 262144 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-27b | chat | 262144 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-35b-a3b | chat | 262144 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-397b-a17b | chat | 262144 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-flash-02-23 | chat | 1000000 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/qwen/qwen3.5-plus-02-15 | chat | 1000000 | 65536 | Y | Y | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/sentence-transformers/all-minilm-l12-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/sentence-transformers/all-minilm-l6-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/sentence-transformers/all-mpnet-base-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/sentence-transformers/multi-qa-mpnet-base-dot-v1 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/sentence-transformers/paraphrase-minilm-l6-v2 | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/switchpoint/router | chat | 131072 | 131072 | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/thenlper/gte-base | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/thenlper/gte-large | embedding | 512 | 512 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/undi95/remm-slerp-l2-13b | chat | 6144 | - | - | - | - | - | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/x-ai/grok-4 | chat | 256000 | 256000 | Y | - | Y | - | Y | - | - | - | - | - | - | - | - | - | - | - | Y | - | - | - |
| openrouter/xiaomi/mimo-v2-flash | chat | 262144 | 16384 | Y | N | Y | N | Y | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| openrouter/z-ai/glm-4.6 | chat | 202800 | 131000 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |
| openrouter/z-ai/glm-4.6:exacto | chat | 202800 | 131000 | Y | - | Y | Y | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |
| openrouter/z-ai/glm-4.7 | chat | 202752 | 64000 | Y | Y | Y | N | Y | - | - | - | - | - | - | - | - | Y | Y | - | - | - | - | - |
| openrouter/z-ai/glm-4.7-flash | chat | 200000 | 32000 | Y | Y | Y | N | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |
| openrouter/z-ai/glm-5 | chat | 202752 | 128000 | Y | - | Y | - | Y | - | - | - | - | - | - | - | - | Y | - | - | - | - | - | - |

<!-- END openrouter-capability-table (generated) -->
