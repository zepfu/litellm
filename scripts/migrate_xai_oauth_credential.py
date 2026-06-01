#!/usr/bin/env python3
"""Migrate Hermes xAI OAuth credentials into a LiteLLM-owned auth file."""

from __future__ import annotations

import argparse
from pathlib import Path

from litellm.llms.xai.oauth import migrate_hermes_xai_oauth_credential


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the xai-oauth credential from a Hermes auth.json file into "
            "LiteLLM's managed xAI OAuth credential format."
        )
    )
    parser.add_argument(
        "--hermes-auth-file",
        type=Path,
        default=None,
        help="Path to Hermes auth.json. Defaults to ~/.hermes/auth.json.",
    )
    parser.add_argument(
        "--target-auth-file",
        type=Path,
        default=None,
        help="LiteLLM-owned target path. Defaults to ~/.litellm/xai/oauth-auth.json.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="OAuth scope key for the generated managed credential record.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the LiteLLM-owned target file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    migrated_path = migrate_hermes_xai_oauth_credential(
        hermes_auth_file=args.hermes_auth_file,
        target_auth_file=args.target_auth_file,
        scope=args.scope,
        overwrite=args.overwrite,
    )
    print(f"Migrated xAI OAuth credential to {migrated_path}")  # noqa: T201
    print(  # noqa: T201
        "Set LITELLM_XAI_OAUTH_AUTH_FILE to this path for oa_xai/* routes."
    )


if __name__ == "__main__":
    main()
