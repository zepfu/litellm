#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _decode_jwt_claims_without_validation(token: str) -> dict[str, Any]:
    try:
        parts = token.split('.')
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1]
        payload_b64 += '=' * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
    except Exception:
        return {}


def _extract_account_id_from_token(token: str | None) -> str | None:
    if not token:
        return None
    claims = _decode_jwt_claims_without_validation(token)
    auth_claims = claims.get('https://api.openai.com/auth')
    if isinstance(auth_claims, dict):
        return _clean_str(auth_claims.get('chatgpt_account_id'))
    return None


def _load_codex_forward_headers(auth_path: pathlib.Path) -> dict[str, str]:
    auth_data = json.loads(auth_path.read_text(encoding='utf-8'))
    token_data = auth_data.get('tokens')
    if not isinstance(token_data, dict):
        token_data = auth_data

    access_token = _clean_str(token_data.get('access_token'))
    if access_token is None:
        raise RuntimeError(f'missing access_token in {auth_path}')

    account_id = _clean_str(token_data.get('account_id')) or _extract_account_id_from_token(
        _clean_str(token_data.get('id_token')) or access_token
    )
    if account_id is None:
        raise RuntimeError(f'missing account_id in {auth_path}')

    return {
        'x-pass-authorization': f'Bearer {access_token}',
        'x-pass-chatgpt-account-id': account_id,
        'x-pass-originator': 'codex_cli_rs',
    }


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode('utf-8')
            status_code = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'adapter smoke request failed with HTTP {exc.code}: {body[:400]}') from exc

    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'adapter smoke response was not valid JSON: {body[:400]}') from exc
    return status_code, parsed_body


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a minimal Anthropic-route adapter smoke request.')
    parser.add_argument('--model', required=True, help='Adapted model name, e.g. gpt-5.4')
    parser.add_argument('--base-url', default='http://127.0.0.1:4001/anthropic', help='Anthropic-compatible base URL')
    parser.add_argument('--auth-file', default=str(pathlib.Path.home() / '.codex' / 'auth.json'), help='Path to Codex auth.json for forwarded x-pass headers')
    parser.add_argument('--prompt', default='Reply with exactly two words: adapter smoke', help='User prompt to send')
    parser.add_argument('--user-id', default='litellm.adapter-harness', help='Metadata user_id to stamp into the request')
    parser.add_argument('--timeout-seconds', type=float, default=60.0, help='HTTP timeout for the smoke request')
    args = parser.parse_args()

    auth_path = pathlib.Path(args.auth_file).expanduser()
    if not auth_path.exists():
        raise SystemExit(f'Codex auth file not found: {auth_path}')

    unique_seed = f"{args.model}-{time.time()}".encode('utf-8')
    unique_suffix = __import__('hashlib').sha256(unique_seed).hexdigest()
    session_id = f'adapter-{args.model}-{unique_suffix[:12]}'
    prompt_marker = f'adapter-marker-{args.model}-{unique_suffix[12:20]}'
    request_payload = {
        'model': args.model,
        'max_tokens': 64,
        'messages': [
            {
                'role': 'user',
                'content': f"{args.prompt}\nMarker: {prompt_marker}",
            }
        ],
        'metadata': {
            'session_id': session_id,
            'user_id': args.user_id,
        },
    }
    headers = {
        'content-type': 'application/json',
        'x-api-key': 'test-key',
        **_load_codex_forward_headers(auth_path),
    }

    started = time.time()
    status_code, response_body = _post_json(
        f"{args.base_url.rstrip('/')}/v1/messages?beta=true",
        headers=headers,
        payload=request_payload,
        timeout=args.timeout_seconds,
    )
    duration_seconds = round(time.time() - started, 3)

    if status_code != 200:
        raise RuntimeError(f'unexpected HTTP status: {status_code}')
    if response_body.get('type') != 'message':
        raise RuntimeError(f"unexpected response type: {response_body.get('type')!r}")
    if response_body.get('model') != args.model:
        raise RuntimeError(f"unexpected response model: {response_body.get('model')!r}")

    content = response_body.get('content')
    if not isinstance(content, list) or not content:
        raise RuntimeError('response content missing')
    first_block = content[0]
    if not isinstance(first_block, dict) or first_block.get('type') != 'text':
        raise RuntimeError(f'unexpected first content block: {first_block!r}')
    text = _clean_str(first_block.get('text'))
    if text is None:
        raise RuntimeError('response text missing')

    usage = response_body.get('usage')
    if not isinstance(usage, dict):
        raise RuntimeError('response usage missing')

    print(json.dumps({
        'session_id': session_id,
        'prompt_marker': prompt_marker,
        'model': args.model,
        'response_text': text,
        'duration_seconds': duration_seconds,
        'usage': usage,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
