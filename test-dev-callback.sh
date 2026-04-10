#!/usr/bin/env bash
# Test AawmAgentIdentity callback enrichment in dev container
set -euo pipefail

PORT="${1:-4002}"
BASE="http://localhost:${PORT}"

echo "=== Test 1: Subagent request (identity in first user message) ==="
echo "Sending POST to ${BASE}/anthropic/v1/messages ..."

HTTP_CODE=$(curl -s -o /tmp/dev-test-response.json -w '%{http_code}' \
  "${BASE}/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-fake-key-for-testing" \
  -H "anthropic-version: 2023-06-01" \
  -H "langfuse_trace_name: claude-code" \
  -X POST \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "system": "You are a Claude agent, built on Anthropic'\''s Claude Agent SDK.",
    "messages": [
      {
        "role": "user",
        "content": "You are '\''engineer'\'' and you are working on the '\''litellm'\'' project.\n\nPlease say hello."
      }
    ]
  }')

echo "HTTP status: ${HTTP_CODE}"
echo "Response: $(cat /tmp/dev-test-response.json | head -5)"
echo ""

echo "=== Test 2: Orchestrator request (no identity pattern) ==="
HTTP_CODE2=$(curl -s -o /tmp/dev-test-response2.json -w '%{http_code}' \
  "${BASE}/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-fake-key-for-testing" \
  -H "anthropic-version: 2023-06-01" \
  -H "langfuse_trace_name: claude-code" \
  -X POST \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "messages": [
      {
        "role": "user",
        "content": "Hello, please respond briefly."
      }
    ]
  }')

echo "HTTP status: ${HTTP_CODE2}"
echo "Response: $(cat /tmp/dev-test-response2.json | head -5)"
echo ""

echo "=== Checking container logs for enrichment ==="
echo "Looking for AawmAgentIdentity log lines..."
docker logs litellm-dev 2>&1 | grep -i "AawmAgentIdentity" | tail -20
echo ""
echo "=== Done ==="
