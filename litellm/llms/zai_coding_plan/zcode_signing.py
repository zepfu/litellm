"""Conditional ZCode request signing for Coding Plan egress.

Feature-gate and handshake failures stay unsigned. Fail-closed errors never
include credentials, signatures, nonces, ciphers, or session ids.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
import uuid
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import load_der_private_key

_KDF_SALT = b"WD_CLIENT_SIGN_KDF_SALT"
_HANDSHAKE_KDF_INFO = b"getSignKey_hmac"
_PRIVATE_KEY_KDF_INFO = b"ed25519_priv"
_APP_ID = "zcode"
_POW_LEADING_ZERO_BITS = 8
_POW_SEARCH_LIMIT = 1_000_000
_SEPARATOR_MESSAGE = "Client signing credential must contain one separator."
_SIGNING_FAILED_MESSAGE = "Client request signing failed."
_POW_FAILED_MESSAGE = "Client request proof-of-work failed."
_REFRESHABLE_REASONS = frozenset(
    {
        "VERIFY_SIGNATURE_INVALID",
        "VERIFY_APIKEY_EXPIRED",
    }
)

GateTransport = Callable[[str, Mapping[str, str], float], tuple[int, object]]
HandshakeTransport = Callable[
    [str, Mapping[str, str], Mapping[str, object], float],
    tuple[int, object],
]

_STATE_LOCK = threading.Lock()
_gate_cache: dict[tuple[str, str], tuple[bool, int]] = {}
_private_keys: dict[tuple[str, str], Ed25519PrivateKey] = {}
_rejection_counts: dict[str, int] = {}
_bypass_ids: set[str] = set()


class ZCodeSigningError(Exception):
    """Fail-closed signing error whose message contains no secret material."""

    def __init__(self, message: str, *, fail_open: bool = False) -> None:
        super().__init__(message)
        self.fail_open = fail_open


def classify_refreshable_signature_reason(payload: object) -> Optional[str]:
    """Return an exact refreshable signing reason, or None."""

    try:
        return _match_refreshable_reason(payload)
    except Exception:
        return None


def attach_zcode_signature_headers(
    headers: Mapping[str, str],
    *,
    api_key: str,
    runtime_header_version: str,
    feature_gate_url: str,
    handshake_url: str,
    feature_gate_headers: Optional[Mapping[str, str]] = None,
    feature_gate_timeout_seconds: float = 15,
    feature_gate_cache_ttl_seconds: float = 3600,
    handshake_timeout_seconds: float = 10,
    gate_transport: Optional[GateTransport] = None,
    handshake_transport: Optional[HandshakeTransport] = None,
    now_ms: Optional[Callable[[], int]] = None,
    random_bytes: Optional[Callable[[int], bytes]] = None,
) -> dict[str, str]:
    """Return a new header dict, with signing headers only when the gate is enabled."""

    unsigned = _copy_headers(headers)
    cache_id = _api_key_cache_id(api_key)
    if _bypass_active(cache_id):
        return unsigned
    if not _feature_gate_enabled(
        api_key=api_key,
        cache_id=cache_id,
        feature_gate_url=feature_gate_url,
        feature_gate_headers=feature_gate_headers,
        feature_gate_timeout_seconds=feature_gate_timeout_seconds,
        feature_gate_cache_ttl_seconds=feature_gate_cache_ttl_seconds,
        gate_transport=gate_transport,
        now_ms=now_ms,
    ):
        return unsigned
    api_key_id, api_key_secret = _split_client_credential(api_key)
    session_id = _required_session_id(unsigned)
    private_key = _load_private_key(
        api_key=api_key,
        api_key_id=api_key_id,
        api_key_secret=api_key_secret,
        cache_id=cache_id,
        handshake_url=handshake_url,
        handshake_timeout_seconds=handshake_timeout_seconds,
        handshake_transport=handshake_transport,
        now_ms=now_ms,
        random_bytes=random_bytes,
    )
    if private_key is None or _bypass_active(cache_id):
        return unsigned
    try:
        return _signed_request_headers(
            unsigned,
            api_key_id=api_key_id,
            session_id=session_id,
            runtime_header_version=runtime_header_version,
            private_key=private_key,
            now_ms=now_ms,
            random_bytes=random_bytes,
        )
    except ZCodeSigningError:
        raise
    except Exception:
        raise ZCodeSigningError(_SIGNING_FAILED_MESSAGE, fail_open=False) from None


def note_signature_rejection(api_key: str) -> str:
    """Drop the cached private key. A second rejection bypasses signing."""

    cache_id = _api_key_cache_id(api_key)
    with _STATE_LOCK:
        _drop_private_keys_locked(cache_id)
        if cache_id in _bypass_ids:
            return "bypass"
        seen = _rejection_counts.get(cache_id, 0) + 1
        _rejection_counts[cache_id] = seen
        if seen >= 2:
            _bypass_ids.add(cache_id)
            return "bypass"
        return "retry"


def reset_zcode_signing_state() -> None:
    """Clear cached gate decisions, private keys, rejection counts, and bypass."""

    with _STATE_LOCK:
        _gate_cache.clear()
        _private_keys.clear()
        _rejection_counts.clear()
        _bypass_ids.clear()


def has_leading_zero_bits(digest: bytes, bits: int) -> bool:
    """Return whether ``digest`` starts with ``bits`` leading zero bits.

    Whole leading bytes must be ``0x00``. Any remainder is taken from the high
    bits of the next byte, so eight bits requires the first byte to be ``0x00``.
    """

    if bits <= 0:
        return True
    full_bytes, remaining_bits = divmod(bits, 8)
    needed = full_bytes + (1 if remaining_bits else 0)
    if len(digest) < needed:
        return False
    if digest[:full_bytes] != b"\x00" * full_bytes:
        return False
    if remaining_bits == 0:
        return True
    mask = (0xFF << (8 - remaining_bits)) & 0xFF
    return (digest[full_bytes] & mask) == 0


def _match_refreshable_reason(payload: object) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("msg", "reason"):
        matched = _refreshable_text(payload.get(key))
        if matched is not None:
            return matched
    data = payload.get("data")
    if isinstance(data, dict):
        matched = _refreshable_text(data.get("reason"))
        if matched is not None:
            return matched
    error = payload.get("error")
    if isinstance(error, dict):
        for key in ("reason", "message"):
            matched = _refreshable_text(error.get(key))
            if matched is not None:
                return matched
    return None


def _refreshable_text(value: object) -> Optional[str]:
    if isinstance(value, str) and value in _REFRESHABLE_REASONS:
        return value
    return None


def _feature_gate_enabled(
    *,
    api_key: str,
    cache_id: str,
    feature_gate_url: str,
    feature_gate_headers: Optional[Mapping[str, str]],
    feature_gate_timeout_seconds: float,
    feature_gate_cache_ttl_seconds: float,
    gate_transport: Optional[GateTransport],
    now_ms: Optional[Callable[[], int]],
) -> bool:
    cache_key = (cache_id, feature_gate_url)
    cached = _read_gate_cache(cache_key, _current_ms(now_ms))
    if cached is not None:
        return cached
    transport = gate_transport or _default_gate_transport
    try:
        status, payload = transport(
            feature_gate_url,
            _gate_request_headers(feature_gate_headers, api_key),
            feature_gate_timeout_seconds,
        )
        decision = _interpret_feature_gate(status, payload)
    except Exception:
        return False
    if decision is None:
        return False
    _write_gate_cache(
        cache_key,
        enabled=decision,
        now=_current_ms(now_ms),
        ttl_seconds=feature_gate_cache_ttl_seconds,
    )
    return decision


def _interpret_feature_gate(status: object, payload: object) -> Optional[bool]:
    """Return a cacheable enabled flag, or None when the response must not be cached."""

    if status != 200 or not isinstance(payload, dict):
        return None
    if payload.get("code") != 0:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    if "codingPlanSignature" not in data:
        return False
    signature_config = data.get("codingPlanSignature")
    if isinstance(signature_config, dict) and signature_config.get("enable") is True:
        return True
    return False


def _split_client_credential(api_key: str) -> tuple[str, str]:
    if api_key.count(".") != 1:
        raise ZCodeSigningError(_SEPARATOR_MESSAGE, fail_open=False)
    api_key_id, api_key_secret = api_key.split(".", 1)
    if not api_key_id.strip() or not api_key_secret.strip():
        raise ZCodeSigningError(_SEPARATOR_MESSAGE, fail_open=False)
    return api_key_id, api_key_secret


def _required_session_id(headers: Mapping[str, str]) -> str:
    session_id = _session_id_from_headers(headers)
    if session_id is None:
        return str(uuid.uuid4())
    return session_id


def _session_id_from_headers(headers: Mapping[str, str]) -> Optional[str]:
    for key in ("X-Session-Id", "x-session-id"):
        value = headers.get(key)
        if isinstance(value, str) and value != "":
            return value
    return None


def _load_private_key(
    *,
    api_key: str,
    api_key_id: str,
    api_key_secret: str,
    cache_id: str,
    handshake_url: str,
    handshake_timeout_seconds: float,
    handshake_transport: Optional[HandshakeTransport],
    now_ms: Optional[Callable[[], int]],
    random_bytes: Optional[Callable[[int], bytes]],
) -> Optional[Ed25519PrivateKey]:
    cache_key = (cache_id, _handshake_origin(handshake_url))
    with _STATE_LOCK:
        cached = _private_keys.get(cache_key)
    if cached is not None:
        return cached
    imported = _handshake_for_private_key(
        api_key=api_key,
        api_key_id=api_key_id,
        api_key_secret=api_key_secret,
        handshake_url=handshake_url,
        handshake_timeout_seconds=handshake_timeout_seconds,
        handshake_transport=handshake_transport,
        now_ms=now_ms,
        random_bytes=random_bytes,
    )
    if imported is None:
        return None
    with _STATE_LOCK:
        if cache_id in _bypass_ids:
            return None
        _private_keys[cache_key] = imported
    return imported


def _handshake_for_private_key(
    *,
    api_key: str,
    api_key_id: str,
    api_key_secret: str,
    handshake_url: str,
    handshake_timeout_seconds: float,
    handshake_transport: Optional[HandshakeTransport],
    now_ms: Optional[Callable[[], int]],
    random_bytes: Optional[Callable[[int], bytes]],
) -> Optional[Ed25519PrivateKey]:
    source = _random_source(random_bytes)
    try:
        timestamp_ms = _current_ms(now_ms)
        nonce = source(16).hex()
        signature = _handshake_signature(
            api_key_id=api_key_id,
            api_key_secret=api_key_secret,
            timestamp=str(timestamp_ms),
            nonce=nonce,
        )
    except Exception:
        return None
    transport = handshake_transport or _default_handshake_transport
    try:
        status, payload = transport(
            handshake_url,
            {
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
            {
                "apiKey": api_key,
                "nonce": nonce,
                "sig": signature,
                "ts": str(timestamp_ms),
            },
            handshake_timeout_seconds,
        )
    except Exception:
        return None
    private_cipher = _private_cipher(status, payload)
    if private_cipher is None:
        return None
    try:
        return _decrypt_private_key(private_cipher, api_key_id, api_key_secret)
    except Exception:
        return None


def _private_cipher(status: object, payload: object) -> Optional[str]:
    if status != 200 or not isinstance(payload, dict):
        return None
    if "code" in payload and payload.get("code") != 200:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    private_cipher = data.get("privateCipher")
    if not isinstance(private_cipher, str):
        return None
    return private_cipher


def _handshake_signature(
    *,
    api_key_id: str,
    api_key_secret: str,
    timestamp: str,
    nonce: str,
) -> str:
    mac = HMAC(_hkdf_sha256(api_key_secret, _HANDSHAKE_KDF_INFO), hashes.SHA256())
    message = f"get_sign_key\n{api_key_id}\n{timestamp}\n{nonce}".encode("utf-8")
    mac.update(message)
    return base64.b64encode(mac.finalize()).decode("ascii")


def _decrypt_private_key(
    private_cipher: str,
    api_key_id: str,
    api_key_secret: str,
) -> Ed25519PrivateKey:
    decoded = base64.b64decode("".join(private_cipher.split()), validate=True)
    if len(decoded) < 12 + 16:
        raise ValueError("truncated private cipher")
    plaintext = AESGCM(_hkdf_sha256(api_key_secret, _PRIVATE_KEY_KDF_INFO)).decrypt(
        decoded[:12],
        decoded[12:],
        api_key_id.encode("utf-8"),
    )
    pkcs8 = base64.b64decode("".join(plaintext.decode("utf-8").split()), validate=True)
    private_key = load_der_private_key(pkcs8, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("unexpected private key type")
    return private_key


def _signed_request_headers(
    headers: Mapping[str, str],
    *,
    api_key_id: str,
    session_id: str,
    runtime_header_version: str,
    private_key: Ed25519PrivateKey,
    now_ms: Optional[Callable[[], int]],
    random_bytes: Optional[Callable[[int], bytes]],
) -> dict[str, str]:
    source = _random_source(random_bytes)
    timestamp = str(_current_ms(now_ms))
    client_nonce = source(16).hex()
    client_version = (
        runtime_header_version
        if isinstance(runtime_header_version, str)
        else str(runtime_header_version)
    )
    signature = _request_signature(
        private_key,
        api_key_id=api_key_id,
        timestamp=timestamp,
        client_version=client_version,
        session_id=session_id,
        client_nonce=client_nonce,
    )
    proof = _proof_of_work(
        api_key_id=api_key_id,
        session_id=session_id,
        timestamp=timestamp,
        random_bytes=source,
    )
    retained = {key: value for key, value in headers.items() if key != "x-session-id"}
    return {
        **retained,
        "X-Client-Ts": timestamp,
        "X-Client-Version": client_version,
        "X-Client-Nonce": client_nonce,
        "X-Client-Sig": signature,
        "X-Session-Id": session_id,
        "X-App-Id": _APP_ID,
        "X-Client-Pow": proof,
    }


def _request_signature(
    private_key: Ed25519PrivateKey,
    *,
    api_key_id: str,
    timestamp: str,
    client_version: str,
    session_id: str,
    client_nonce: str,
) -> str:
    message = f"{api_key_id}\n{timestamp}\n{client_version}\n{session_id}\n{client_nonce}".encode(
        "utf-8"
    )
    try:
        raw_signature = private_key.sign(message)
    except Exception:
        raise ZCodeSigningError(_SIGNING_FAILED_MESSAGE, fail_open=False) from None
    return base64.b64encode(raw_signature).decode("ascii")


def _proof_of_work(
    *,
    api_key_id: str,
    session_id: str,
    timestamp: str,
    random_bytes: Callable[[int], bytes],
) -> str:
    seed_material = f"{api_key_id}\n{_APP_ID}\n{session_id}\n{timestamp}".encode(
        "utf-8"
    )
    pow_seed = hashlib.sha256(seed_material).hexdigest()[:32]
    pow_prefix = random_bytes(12).hex()
    for counter in range(_POW_SEARCH_LIMIT):
        pow_value = f"{pow_prefix}{counter:08x}"
        digest = hashlib.sha256(f"{pow_seed}\n{pow_value}".encode("utf-8")).digest()
        if has_leading_zero_bits(digest, _POW_LEADING_ZERO_BITS):
            return pow_value
    raise ZCodeSigningError(_POW_FAILED_MESSAGE, fail_open=False)


def _hkdf_sha256(secret: str, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_KDF_SALT,
        info=info,
    ).derive(secret.encode("utf-8"))


def _default_gate_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, object]:
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        response = client.get(url, headers=dict(headers))
        return response.status_code, response.json()


def _default_handshake_transport(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, object],
    timeout: float,
) -> tuple[int, object]:
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        response = client.post(url, headers=dict(headers), json=dict(body))
        return response.status_code, response.json()


def _read_gate_cache(cache_key: tuple[str, str], now: int) -> Optional[bool]:
    with _STATE_LOCK:
        entry = _gate_cache.get(cache_key)
        if entry is None:
            return None
        enabled, expires_at = entry
        if now >= expires_at:
            del _gate_cache[cache_key]
            return None
        return enabled


def _write_gate_cache(
    cache_key: tuple[str, str],
    *,
    enabled: bool,
    now: int,
    ttl_seconds: float,
) -> None:
    expires_at = now + int(ttl_seconds * 1000)
    with _STATE_LOCK:
        _gate_cache[cache_key] = (enabled, expires_at)


def _bypass_active(cache_id: str) -> bool:
    with _STATE_LOCK:
        return cache_id in _bypass_ids


def _drop_private_keys_locked(cache_id: str) -> None:
    stale = [key for key in _private_keys if key[0] == cache_id]
    for key in stale:
        del _private_keys[key]


def _api_key_cache_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _handshake_origin(handshake_url: str) -> str:
    parts = urlsplit(handshake_url)
    if not parts.scheme or not parts.netloc:
        return handshake_url
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return f"{parts.scheme}://{netloc}"


def _current_ms(now_ms: Optional[Callable[[], int]]) -> int:
    if now_ms is None:
        return int(time.time() * 1000)
    return int(now_ms())


def _random_source(
    random_bytes: Optional[Callable[[int], bytes]],
) -> Callable[[int], bytes]:
    if random_bytes is None:
        return secrets.token_bytes
    return random_bytes


def _copy_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items()}


def _gate_request_headers(
    feature_gate_headers: Optional[Mapping[str, str]],
    api_key: str,
) -> dict[str, str]:
    supplied = feature_gate_headers or {}
    return {
        **{key: value for key, value in supplied.items() if key.lower() != "x-api-key"},
        "x-api-key": api_key,
    }
