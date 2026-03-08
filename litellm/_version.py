import importlib_metadata

try:
    version = importlib_metadata.version("litellm")
except Exception:
    version = "unknown"

# AAWM fork: ensure the health endpoint reports the full fork version
# (e.g. "1.82.1-aawm.4").  The PEP 440 local label (+aawm.4) may or may
# not be retained by pip depending on the Python version, so we normalise
# to the hyphenated form unconditionally.
import re as _re
version = _re.sub(r"[+]aawm\.\d+$", "", version)
if "aawm" not in version:
    version = f"{version}-aawm.4"
