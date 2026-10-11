# Debugging

2 levels of debugging supported. 

- debug (prints info logs)
- detailed debug (prints debug logs)

The proxy also supports json logs. [See here](#json-logs)

## `debug`

**via cli**

```bash showLineNumbers
$ litellm --debug
```

**via env**

```python showLineNumbers
os.environ["LITELLM_LOG"] = "INFO"
```

## `detailed debug`

**via cli**

```bash showLineNumbers
$ litellm --detailed_debug
```

**via env**

```python showLineNumbers
os.environ["LITELLM_LOG"] = "DEBUG"
```

### Debug Logs 

Run the proxy with `--detailed_debug` to view detailed debug logs
```shell showLineNumbers
litellm --config /path/to/config.yaml --detailed_debug
```

When making requests you should see the POST request sent by LiteLLM to the LLM on the Terminal output
```shell showLineNumbers
POST Request Sent from LiteLLM:
curl -X POST \
https://api.openai.com/v1/chat/completions \
-H 'content-type: application/json' -H 'Authorization: Bearer sk-qnWGUIW9****************************************' \
-d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "this is a test request, write a short poem"}]}'
```

## Debug single request

Pass in `litellm_request_debug=True` in the request body

```bash showLineNumbers
curl -L -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "model":"fake-openai-endpoint",
    "messages": [{"role": "user","content": "How many r in the word strawberry?"}],
    "litellm_request_debug": true
}'
```

This will emit the raw request sent by LiteLLM to the API Provider and raw response received from the API Provider for **just** this request in the logs. 


```bash showLineNumbers
INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
20:14:06 - LiteLLM:WARNING: litellm_logging.py:938 - 

POST Request Sent from LiteLLM:
curl -X POST \
https://exampleopenaiendpoint-production.up.railway.app/chat/completions \
-H 'Authorization: Be****ey' -H 'Content-Type: application/json' \
-d '{'model': 'fake', 'messages': [{'role': 'user', 'content': 'How many r in the word strawberry?'}], 'stream': False}'


20:14:06 - LiteLLM:WARNING: litellm_logging.py:1015 - RAW RESPONSE:
{"id":"chatcmpl-817fc08f0d6c451485d571dab39b26a1","object":"chat.completion","created":1677652288,"model":"gpt-3.5-turbo-0301","system_fingerprint":"fp_44709d6fcb","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nHello there, how may I assist you today?"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}


INFO:     127.0.0.1:56155 - "POST /chat/completions HTTP/1.1" 200 OK

```


## JSON LOGS

Set `JSON_LOGS="True"` in your env:

```bash showLineNumbers
export JSON_LOGS="True"
```
**OR**

Set `json_logs: true` in your yaml: 

```yaml showLineNumbers
litellm_settings:
    json_logs: true
```

Start proxy 

```bash showLineNumbers
$ litellm
```

The proxy will now all logs in json format.

## Proxy lifecycle diagnostics

Proxy startup, configuration application, explicit initialization, and
shutdown emit bounded `proxy_lifecycle` records. These records contain process
and parent PIDs, a lifecycle counter, process start ticks where available, a
container instance ID where available, and a UTC Unix timestamp in nanoseconds.
Successful config application includes an ephemeral keyed
`config_revision_id` computed from the effective in-memory config; only the
opaque identifier is logged. It is not a durable cross-restart version number.
For local files, `config_mtime_ns`,
`config_size_bytes`, and `config_inode` provide additional content-free source
metadata. Configuration paths and values are not included. Lifecycle records
are capped at 20 per process per second; `suppressed_events` reports records
skipped since the previous emitted record.

`signal_observer_status=installed` means the proxy wrapped a recognized
Uvicorn, Gunicorn, or Hypercorn `SIGINT`/`SIGTERM` handler and delegates each
signal to that handler unchanged. `proxy_shutdown` includes the observed signal
number when available. A zero signal number means the proxy did not observe
either signal through a recognized handler.

Identify an in-process lifespan re-entry from repeated
`proxy_lifespan_start` records with the same PID and process start ticks and
increasing `lifecycle_id` values. `initialize_started` and
`initialize_completed` describe explicit initialization calls and do not by
themselves establish lifespan re-entry. Changed process identity indicates a
new process. Correlate process and container IDs with the container runtime's
start, restart, and recreate events to distinguish process replacement from a
container restart; the proxy cannot infer that distinction from its own logs.

## Control Log Output 

Turn off fastapi's default 'INFO' logs 

1. Turn on 'json logs' 
```yaml showLineNumbers
litellm_settings:
    json_logs: true
```

2. Set `LITELLM_LOG` to 'ERROR' 

Only get logs if an error occurs. 

```bash showLineNumbers
LITELLM_LOG="ERROR"
```

3. Start proxy 


```bash showLineNumbers
$ litellm
```

Expected Output: 

```bash showLineNumbers
# no info statements
```

## Common Errors 

1. "No available deployments..."

```
No deployments available for selected model, Try again in 60 seconds. Passed model=claude-3-5-sonnet. pre-call-checks=False, allowed_model_region=n/a.
```

This can be caused due to all your models hitting rate limit errors, causing the cooldown to kick in. 

How to control this? 
- Adjust the cooldown time

```yaml showLineNumbers
router_settings:
    cooldown_time: 0 # 👈 KEY CHANGE
```

- Disable Cooldowns [NOT RECOMMENDED]

```yaml showLineNumbers
router_settings:
    disable_cooldowns: True
```

This is not recommended, as it will lead to requests being routed to deployments over their tpm/rpm limit.
