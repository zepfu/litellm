"""Shared credential_error_sanitizer (RR-065/074/075/092)."""

from __future__ import annotations

from litellm.secret_managers.credential_error_sanitizer import (
    DEFAULT_SECRET_FIELD_NAMES,
    SECRET_FIELD_NAMES,
    sanitize_credential_error_message,
)


def test_default_secret_field_names_cover_oauth_tokens() -> None:
    assert SECRET_FIELD_NAMES is DEFAULT_SECRET_FIELD_NAMES
    assert {
        "access_token",
        "client_secret",
        "id_token",
        "key",
        "refresh_token",
    } <= set(DEFAULT_SECRET_FIELD_NAMES)


def test_sanitize_redacts_secret_values_not_only_labels() -> None:
    raw = (
        "invalid_grant: access_token=eyJhbGciOi.live.token.value "
        "refresh_token: rt-super-secret "
        "client_secret=cs-xyz id_token= id.tok key=k-secret"
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "eyJhbGciOi.live.token.value" not in sanitized
    assert "rt-super-secret" not in sanitized
    assert "cs-xyz" not in sanitized
    assert "id.tok" not in sanitized
    assert "k-secret" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    assert "id_token=[REDACTED]" in sanitized
    assert "key=[REDACTED]" in sanitized
    assert "super-secret" not in sanitized


def test_sanitize_is_case_insensitive_on_field_names() -> None:
    raw = "Access_Token=ABC Refresh_Token: def Client_Secret=ghi"
    sanitized = sanitize_credential_error_message(raw)
    assert "ABC" not in sanitized
    assert "def" not in sanitized
    assert "ghi" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized


def test_sanitize_redacts_double_and_single_quoted_values_without_partial_leak() -> None:
    raw = (
        'access_token="eyJ.double.quoted.secret" '
        "refresh_token='rt-single-quoted-secret' "
        'client_secret="cs-with-\\"escape\\"-secret"'
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "eyJ.double.quoted.secret" not in sanitized
    assert "rt-single-quoted-secret" not in sanitized
    assert "cs-with" not in sanitized
    assert "escape" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]]") == 0
    # No trailing quote/suffix leakage from partial match.
    assert '="[' not in sanitized
    assert "=['" not in sanitized


def test_sanitize_redacts_json_object_member_shapes_provider_like() -> None:
    # Google / OpenAI-style error bodies often embed JSON fragments.
    raw = (
        'invalid_grant: {"error":"invalid_grant","error_description":'
        '"Token is expired","access_token":"ya29.a0AfH6SMBx-SECRET",'
        '"refresh_token":"1//0g-REFRESH-SECRET","client_secret":"GOCSPX-SECRET"}'
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "ya29.a0AfH6SMBx-SECRET" not in sanitized
    assert "1//0g-REFRESH-SECRET" not in sanitized
    assert "GOCSPX-SECRET" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]]") == 0
    # Non-secret JSON keys remain useful for operators.
    assert "invalid_grant" in sanitized


def test_sanitize_redacts_json_with_spaces_and_single_quoted_keys() -> None:
    raw = (
        "{ 'access_token' : 'sq-json-token' , "
        "\"refresh_token\" : \"dq-json-token\" }"
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "sq-json-token" not in sanitized
    assert "dq-json-token" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized


def test_sanitize_redacts_authorization_bearer_scoped_only() -> None:
    raw = (
        "upstream 401 Authorization: Bearer eyJhbGciOi.bearer.secret "
        "note bearer of bad news without header should stay"
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "eyJhbGciOi.bearer.secret" not in sanitized
    assert "Authorization: Bearer [REDACTED]" in sanitized
    # English phrase containing "bearer" is not treated as a credential.
    assert "bearer of bad news" in sanitized


def test_sanitize_can_disable_authorization_bearer_redaction() -> None:
    raw = "Authorization: Bearer keep-me"
    out = sanitize_credential_error_message(
        raw, redact_authorization_bearer=False
    )
    assert "keep-me" in out


def test_sanitize_does_not_match_key_as_substring() -> None:
    raw = "api_key=should-not-match-key-alone foo_key=bar"
    out = sanitize_credential_error_message(raw)
    assert out == raw


def test_sanitize_truncates_when_limit_set() -> None:
    long = "x" * 2000
    out = sanitize_credential_error_message(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")


def test_sanitize_without_limit_preserves_length() -> None:
    long = "plain text " + ("y" * 600)
    out = sanitize_credential_error_message(long)
    assert out == long


def test_sanitize_custom_field_names_only() -> None:
    raw = "access_token=keep-me custom_secret=hide-me"
    out = sanitize_credential_error_message(
        raw, field_names=("custom_secret",)
    )
    assert "keep-me" in out
    assert "hide-me" not in out
    assert "custom_secret=[REDACTED]" in out


def test_sanitize_redacts_even_backslash_quoted_suffix_leaks() -> None:
    """Even-backslash closes early; trailing quoted suffix must not leak."""
    # Two backslashes: naive consumers close after \\" then leave suffix".
    raw_even = r'access_token="secret\\"suffix"'
    sanitized = sanitize_credential_error_message(raw_even)
    assert "secret" not in sanitized
    assert "suffix" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert sanitized.count("[REDACTED]") == 1

    raw_sq = r"access_token='secret\\'suffix'"
    sanitized_sq = sanitize_credential_error_message(raw_sq)
    assert "secret" not in sanitized_sq
    assert "suffix" not in sanitized_sq
    assert "access_token=[REDACTED]" in sanitized_sq

    # Four-backslash variant also must not leave a suffix.
    raw_four = r'access_token="secret\\\\"suffix"'
    out_four = sanitize_credential_error_message(raw_four)
    assert "suffix" not in out_four
    assert "access_token=[REDACTED]" in out_four


def test_sanitize_redacts_escaped_quotes_inside_quoted_values() -> None:
    # Escaped double quote inside the value: access_token="sec\"ret-value"
    raw = r'access_token="sec\"ret-value"'
    sanitized = sanitize_credential_error_message(raw)
    assert "ret-value" not in sanitized
    assert "access_token=[REDACTED]" in sanitized


def test_sanitize_redacts_authorization_bearer_json_dict_and_quoted_assign() -> None:
    cases = [
        (
            '{"Authorization":"Bearer tok-json-secret"}',
            "tok-json-secret",
            '"Bearer [REDACTED]"',
        ),
        (
            "{'Authorization': 'Bearer tok-dict-secret'}",
            "tok-dict-secret",
            "'Bearer [REDACTED]'",
        ),
        (
            'Authorization="Bearer tok-quoted-secret"',
            "tok-quoted-secret",
            '"Bearer [REDACTED]"',
        ),
        (
            "Authorization='Bearer tok-quoted-sq'",
            "tok-quoted-sq",
            "'Bearer [REDACTED]'",
        ),
        (
            "Authorization=Bearer tok-eq-secret",
            "tok-eq-secret",
            "Bearer [REDACTED]",
        ),
        (
            '{ "Authorization" : "Bearer spaced-token" }',
            "spaced-token",
            "Bearer [REDACTED]",
        ),
    ]
    for raw, secret, marker in cases:
        sanitized = sanitize_credential_error_message(raw)
        assert secret not in sanitized, raw
        assert marker in sanitized, (raw, sanitized)


def test_sanitize_redacts_authorization_bearer_even_backslash_quoted_suffix() -> None:
    """Quoted Authorization Bearer even-backslash early-close must not leak."""
    cases = [
        (
            r'Authorization="Bearer secret\\"suffix"',
            ("secret", "suffix"),
            'Authorization="Bearer [REDACTED]"',
        ),
        (
            r"Authorization='Bearer secret\\'suffix'",
            ("secret", "suffix"),
            "Authorization='Bearer [REDACTED]'",
        ),
        (
            r'{"Authorization":"Bearer secret\\"suffix"}',
            ("secret", "suffix"),
            '"Authorization":"Bearer [REDACTED]"',
        ),
        (
            r"{'Authorization': 'Bearer secret\\'suffix'}",
            ("secret", "suffix"),
            "'Authorization': 'Bearer [REDACTED]'",
        ),
        (
            r'Authorization="Bearer secret\\\\"suffix"',
            ("secret", "suffix"),
            'Authorization="Bearer [REDACTED]"',
        ),
        (
            r'Authorization="Bearer sec\\"ret"',
            ("sec", "ret"),
            'Authorization="Bearer [REDACTED]"',
        ),
        # Cross-quote style JSON mixes with even-backslash leak.
        (
            r'''{"Authorization":'Bearer mixed\\'suffix'}''',
            ("mixed", "suffix"),
            '"Authorization":\'Bearer [REDACTED]\'',
        ),
        (
            r"""{'Authorization':"Bearer mixed\\"suffix"}""",
            ("mixed", "suffix"),
            "'Authorization':\"Bearer [REDACTED]\"",
        ),
    ]
    for raw, secrets, marker in cases:
        sanitized = sanitize_credential_error_message(raw)
        for secret in secrets:
            assert secret not in sanitized, (raw, sanitized, secret)
        assert marker in sanitized, (raw, sanitized)
        # No trailing partial-quote leakage of the secret suffix form.
        assert '"suffix"' not in sanitized
        assert "'suffix'" not in sanitized
        assert '"ret"' not in sanitized


def test_sanitize_query_and_form_ampersand_boundaries() -> None:
    raw = (
        "access_token=tok-amp-secret&expires=1&refresh_token=rt-amp-secret"
        "&client_secret=cs-amp-secret&scope=openid"
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "tok-amp-secret" not in sanitized
    assert "rt-amp-secret" not in sanitized
    assert "cs-amp-secret" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    # Adjacent non-secret form fields remain.
    assert "expires=1" in sanitized
    assert "scope=openid" in sanitized
    # No swallowed tail / double-bracket artifacts.
    assert "[REDACTED]]" not in sanitized
    assert sanitized.count("&") >= 2


def test_sanitize_redacts_unquoted_json_secret_values() -> None:
    raw = (
        '{"access_token":ya29.unquoted-secret,'
        '"refresh_token":rt-unquoted,"client_secret":cs-unquoted}'
    )
    sanitized = sanitize_credential_error_message(raw)
    assert "ya29.unquoted-secret" not in sanitized
    assert "rt-unquoted" not in sanitized
    assert "cs-unquoted" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized

    spaced = '{ "access_token" : ya29.space-secret , "x":1}'
    out = sanitize_credential_error_message(spaced)
    assert "ya29.space-secret" not in out
    assert "access_token=[REDACTED]" in out
    assert '"x":1' in out


def test_sanitize_does_not_over_redact_prose_or_key_substrings() -> None:
    prose = (
        "the access_token field was missing from payload; "
        "note bearer of bad news without Authorization header should stay; "
        "api_key=should-not-match-key-alone foo_key=bar"
    )
    out = sanitize_credential_error_message(prose)
    assert out == prose
