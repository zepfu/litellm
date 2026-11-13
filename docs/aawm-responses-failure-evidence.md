# Responses Failure Evidence Capture

The proxy can retain a small private diagnostic for failed OpenAI-compatible
Responses streams. Capture is off by default and independent of Langfuse and
the broader pass-through shape/full-payload captures.

Enable it with `AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE=1`, or create a
process-owned control file containing `1`. The default control path is
`/tmp/captures/responses_failure_evidence.enabled`. A control file is honored
only when it is regular, owned by the proxy process UID, and neither it nor
its parent directory is group/world writable. Set
`AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE_CONTROL_FILE` to select another
control path. Set `AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE_DIR` to select the
output directory; otherwise artifacts go to
`/tmp/captures/responses_failure_evidence`.

For alpha, use the persistent control path
`/app/.analysis/runtime/litellm-alpha/responses-failure-evidence.enabled` and
ensure its parent is persistent, owned by the proxy UID, and mode `0700`.
The host-side parent may create the directory and control file, but the file
must be owned by the proxy UID with mode `0600` and contents `1`; set
`AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE_CONTROL_FILE` to this path. Remove
the file to disable capture.

Each artifact is at most 64 KiB. The writer retains at most 32 artifacts,
uses a mode `0700` directory and atomic mode `0600` files, and deletes older
artifacts when new ones are written. Precommit in-memory event summaries are
limited to eight events. They include event labels, sequence numbers, parser
decisions, bounded field names, response status/ID, and output item types, not
raw SSE.

Artifacts include available request/session correlation IDs, the first
semantic commitment separate from any provisional provider-dispatch latch,
bounded provider-call reservation state, the normalized failure class and
retry classification, and a filtered terminal envelope. The envelope keeps
causal protocol fields such as error type/code/parameter, response ID/status,
and output item types/count. Error messages are hashed. Request bodies,
prompts, user text, headers, credentials, output text, and tool arguments are
not stored. Unknown event names are recorded as `unknown_event` with a
SHA-256 digest. Capture errors do not change request handling.

For a controlled acceptance check after activation, run this inside the alpha
proxy container. It exercises the production artifact writer with a synthetic
typed failure and test request; it does not issue an upstream request:

```bash
AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE=1 \
AAWM_CAPTURE_RESPONSES_FAILURE_EVIDENCE_DIR=/tmp/responses-evidence-check \
python -c 'from starlette.requests import Request; from litellm.proxy.pass_through_endpoints.streaming_handler import ResponsesStreamPreCommitFailure; from litellm.integrations.aawm_passthrough_shape_capture import capture_responses_failure_evidence; r=Request({"type":"http","method":"POST","path":"/v1/responses","headers":[]}); r.state.litellm_call_id="acceptance-check"; f=ResponsesStreamPreCommitFailure(error_class="server_overloaded",classification="transient_capacity",retryable=True,error_code="server_overloaded",error_type="server_error",status_code=529,message="synthetic acceptance"); print(capture_responses_failure_evidence(failure=f,request=r,metadata={},diagnostic={"event_prefix":[{"event_type":"response.failed","sequence_number":1}]}))'
```

Inspect the emitted JSON for the capture kind, synthetic failure class,
correlation ID, private permissions, and absence of request content. This
checks the production writer and output path, not actual live stream capture.
