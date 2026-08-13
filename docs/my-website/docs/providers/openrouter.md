# OpenRouter

## AAWM reference pricing

AAWM keeps OpenRouter pricing route-specific. The maintained
`openrouter/cohere/north-mini-code:free` entry has a third-party hosted
baseline of `$0.20/$0.80` per million input/output tokens sourced from the
NanoGPT and Routeway hosted-model catalog consensus for
`cohere/north-mini-code-1-0`; it is not direct Cohere pricing.
`openrouter/owl-alpha` and the OpenRouter DeepSeek V4 Flash `:free` routes
remain explicitly unpriced when no exact paid-model relationship is verified.

Reference totals are provenance metadata only. They do not populate
`response_cost` or `response_cost_usd`, and a provider-reported cost wins.
LiteLLM can route requests to OpenRouter models through the `openrouter/` prefix.
The [OpenRouter model metadata snapshot](./openrouter-model-metadata) is the
canonical reference for model-specific capability declarations in LiteLLM's
metadata.
Passthrough routing and metadata-backed capability coverage are separate:
uncataloged model IDs can still be sent explicitly to OpenRouter.

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_OpenRouter.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Usage
```python
import os
from litellm import completion

os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_API_BASE"] = "" # [OPTIONAL] defaults to https://openrouter.ai/api/v1
os.environ["OR_SITE_URL"] = "" # [OPTIONAL]
os.environ["OR_APP_NAME"] = "" # [OPTIONAL]

response = completion(
            model="openrouter/google/gemini-2.5-flash",
            messages=messages,
        )
```

## Configuration with Environment Variables

For production environments, you can dynamically configure the base_url using environment variables:

```python
import os
from litellm import completion

# Configure with environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

# Set environment for LiteLLM
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

response = completion(
    model="openrouter/google/gemini-2.5-flash",
    messages=messages,
    base_url=OPENROUTER_BASE_URL  # Explicitly pass base_url for clarity
)
```

This approach provides better flexibility for managing configurations across different environments (dev, staging, production) and makes it easier to switch between self-hosted and cloud endpoints.

## OpenRouter Completion Models
Send `model=openrouter/<your-openrouter-model>` to route a request to
OpenRouter. See OpenRouter's current catalog [here](https://openrouter.ai/models).
For model-specific capability declarations, use the
[OpenRouter model metadata snapshot](./openrouter-model-metadata).

```python
response = completion(
    model="openrouter/<provider>/<model-id>",
    messages=messages,
)
```

### Uncataloged Chat Models

Explicit `openrouter/<model-id>` routing remains available when a model is not
present in LiteLLM's metadata. Metadata-dependent parameter advertisement does
not become universal for these models:

- `reasoning_effort` and `thinking` are advertised only when metadata declares
  `supports_reasoning: true`.
- For `cache_control`, an explicit metadata value of
  `supports_native_cache_control: false` disables forwarding and an explicit
  `true` enables it.
- When that cache-control flag is absent, recognized vendor families may use
  LiteLLM's existing fallback. Unknown models outside those families have
  `cache_control` stripped.

## Passing OpenRouter Params - transforms, models, route
Pass `transforms`, `models`, `route`as arguments to `litellm.completion()`

```python
import os
from litellm import completion

os.environ["OPENROUTER_API_KEY"] = ""

response = completion(
            model="openrouter/google/gemini-2.5-flash",
            messages=messages,
            transforms = [""],
            route= ""
        )
```

## Embedding

OpenRouter embedding requests use the fixed advertised embedding parameter set.
Use a mapped embedding model such as the example below; an uncataloged model ID
may still be passed through to OpenRouter, but it does not add additional
LiteLLM parameters.

```python
from litellm import embedding
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

response = embedding(
    model="openrouter/openai/text-embedding-3-small",
    input=["good morning from litellm", "this is another item"],
)
print(response)
```

## Image Generation

LiteLLM's OpenRouter image-generation adapter converts the standard image
request into OpenRouter's chat-completions request shape. It maps the parameters
documented below and preserves the explicitly routed model ID. Actual image
generation availability and accepted values depend on the selected upstream
OpenRouter model; consult OpenRouter's current model catalog and model details.

### Supported Parameters

- `size`: Maps to OpenRouter's `aspect_ratio` format
  - `1024x1024` → `1:1` (square)
  - `1536x1024` → `3:2` (landscape)
  - `1024x1536` → `2:3` (portrait)
  - `1792x1024` → `16:9` (wide landscape)
  - `1024x1792` → `9:16` (tall portrait)

- `quality`: Maps to OpenRouter's `image_size` format (Gemini models)
  - `low` or `standard` → `1K`
  - `medium` → `2K`
  - `high` or `hd` → `4K`

The adapter performs these mappings for any explicitly routed model. The
selected upstream model must support the resulting `image_size`; `4K` support
is model-specific.

- `n`: Number of images to generate

### Usage

```python
from litellm import image_generation
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Basic image generation
response = image_generation(
    model="openrouter/google/gemini-2.5-flash-image",
    prompt="A beautiful sunset over a calm ocean",
)
print(response)
```

### Advanced Usage with Parameters

```python
from litellm import image_generation
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Generate a 1K landscape image
response = image_generation(
    model="openrouter/google/gemini-2.5-flash-image",
    prompt="A serene mountain landscape with a lake",
    size="1536x1024",  # Landscape format
    quality="standard", # Maps to image_size 1K
)

# Access the generated image
image_data = response.data[0]
if image_data.b64_json:
    # Base64 encoded image
    print(f"Generated base64 image: {image_data.b64_json[:50]}...")
elif image_data.url:
    # Image URL
    print(f"Generated image URL: {image_data.url}")
```

### Using OpenRouter-Specific Parameters

You can also pass OpenRouter-specific parameters directly using `image_config`:

```python
from litellm import image_generation
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

response = image_generation(
    model="openrouter/google/gemini-2.5-flash-image",
    prompt="A futuristic cityscape at night",
    image_config={
        "aspect_ratio": "16:9",  # OpenRouter native format
        "image_size": "1K"       # OpenRouter native format
    }
)
print(response)
```

When `drop_params=False`, image generation may also pass unknown parameters
through to OpenRouter. Whether OpenRouter accepts them remains
model-specific.

### Response Format

The response follows the standard LiteLLM ImageResponse format:

```python
{
    "created": 1703658209,
    "data": [{
        "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",  # Base64 encoded image
        "url": None,
        "revised_prompt": None
    }],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 1290,
        "total_tokens": 1300
    }
}
```

### Cost Tracking

OpenRouter provides cost information in the response, which LiteLLM automatically tracks:

```python
response = image_generation(
    model="openrouter/google/gemini-2.5-flash-image",
    prompt="A cute baby sea otter",
)

# Cost is available in the response metadata
print(f"Request cost: ${response._hidden_params['additional_headers']['llm_provider-x-litellm-response-cost']}")
```

## Image Edit

LiteLLM's OpenRouter image-edit adapter converts the request to OpenRouter's
chat-completions format. It sends the source image as a base64 data URL, adds
the edit prompt as text, and requests `modalities: ["image", "text"]`. The
adapter maps only the recognized parameters documented below. Actual image-edit
availability and accepted values depend on the selected upstream OpenRouter
model; consult OpenRouter's current model catalog and model details.

### Model Availability

See currently available image models and their supported operations on
[OpenRouter's model list](https://openrouter.ai/models?modality=image).

Image edit emits only recognized parameters. An explicitly routed uncataloged
model ID may still reach OpenRouter, but support is determined by OpenRouter and
the selected upstream model.

### Supported Parameters

| Parameter | OpenRouter Mapping | Notes |
|-----------|--------------------|-------|
| `size` | `image_config.aspect_ratio` | `1024x1024` → `1:1`, `1536x1024` → `3:2`, `1024x1536` → `2:3`, `1792x1024` → `16:9`, `1024x1792` → `9:16` |
| `quality` | `image_config.image_size` | `low`/`standard` → `1K`, `medium` → `2K`, `high`/`hd` → `4K` |
| `n` | `n` | Number of images |

:::note
The adapter maps `quality=high` or `quality=hd` to `image_size=4K`, but the
selected upstream model must support 4K output. Compatibility with `2K` and
`4K` is model-specific. `google/gemini-2.5-flash-image` is fixed at `1K`, so
the examples on this page use `quality=standard` (`image_size=1K`).
:::

### Usage

```python
from litellm import image_edit
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Basic image edit
response = image_edit(
    model="openrouter/google/gemini-2.5-flash-image",
    image=open("original_image.png", "rb"),
    prompt="Make the sky a vibrant purple sunset",
)

print(response)
```

### Advanced Usage with Parameters

```python
from litellm import image_edit
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

# Edit with size and quality parameters
response = image_edit(
    model="openrouter/google/gemini-2.5-flash-image",
    image=open("photo.png", "rb"),
    prompt="Add northern lights to the sky",
    size="1536x1024",   # Maps to aspect_ratio 3:2
    quality="standard",  # Maps to image_size 1K
)

# Access the edited image
image_data = response.data[0]
if image_data.b64_json:
    import base64
    with open("edited.png", "wb") as f:
        f.write(base64.b64decode(image_data.b64_json))
```

### Multiple Images Edit

```python
from litellm import image_edit
import os

os.environ["OPENROUTER_API_KEY"] = "your-api-key"

response = image_edit(
    model="openrouter/google/gemini-2.5-flash-image",
    image=[
        open("scene.png", "rb"),
        open("style_reference.png", "rb"),
    ],
    prompt="Blend the reference style into the scene",
)

print(response)
```
