#!/usr/bin/env python3
"""Publish alpha's Kimi identity without updating the CLI or shared descriptor."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

DESCRIPTOR = Path("/home/zepfu/.kimi-code/native-contract-alpha/native-contract.json")
KIMI_BINARY = Path("/home/zepfu/.kimi-code/bin/kimi")
PUBLISHER = Path("/usr/local/sbin/thoth-kimi-update.sh")
TTL_SECONDS = 345600


def main() -> None:
    # Reuse the installed updater's canonical validation and atomic publisher.
    # Only its embedded Python helper runs; the CLI update workflow never runs.
    source = PUBLISHER.read_text()
    marker = "  cat >\"$helper_py\" <<'PY'\n"
    start = source.index(marker) + len(marker)
    helper = source[start : source.index("\nPY\n", start)] + "\n"

    def publish_command(*arguments: str) -> None:
        subprocess.run(
            [sys.executable, "-c", helper, *arguments],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
        )

    publish_command("validate-managed-path", str(KIMI_BINARY), "executable", "755")
    version = subprocess.check_output([str(KIMI_BINARY), "--version"], text=True, timeout=30).strip()
    with tempfile.TemporaryDirectory(prefix=".refresh-", dir=DESCRIPTOR.parent) as stage:
        staged = str(Path(stage) / "native-contract.json")
        publish_command(
            "build-descriptor",
            staged,
            "kimi-code",
            version,
            "kimi-code-cli/" + version,
            str(TTL_SECONDS),
            str(DESCRIPTOR),
        )
        publish_command("validate-descriptor", staged)
        publish_command("promote-descriptor", staged, str(DESCRIPTOR))
    descriptor = json.loads(DESCRIPTOR.read_text())
    sys.stdout.write(
        json.dumps(
            {
                "descriptor": str(DESCRIPTOR),
                "client_version": descriptor["client_version"],
                "expires_at": descriptor["expires_at"],
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
