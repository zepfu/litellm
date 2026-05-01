import os
import sys

from dotenv import load_dotenv

load_dotenv()
import re
from pathlib import Path

repo_base = Path(__file__).resolve().parents[2]

sys.path.insert(
    0, str(repo_base)
)  # Adds the parent directory to the system path
import litellm

public_exceptions = litellm.LITELLM_EXCEPTION_TYPES
# Regular expression to extract the error name
error_name_pattern = re.compile(r"\.exceptions\.([A-Za-z]+Error)")

# Extract error names from each item
error_names = {
    error_name_pattern.search(str(item)).group(1)
    for item in public_exceptions
    if error_name_pattern.search(str(item))
}


# sys.path = original_sys_path


# Parse the documentation to extract documented keys
docs_path = repo_base / "docs/my-website/docs/exception_mapping.md"
documented_keys = set()
try:
    with open(docs_path, "r", encoding="utf-8") as docs_file:
        content = docs_file.read()

        exceptions_section = re.search(
            r"## LiteLLM Exceptions(.*?)\n##", content, re.DOTALL
        )
        if exceptions_section:
            table_content = exceptions_section.group(1)

            exceptions = []
            for line in table_content.splitlines():
                if not line.startswith("|") or "---" in line:
                    continue
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) < 2 or cells[1] == "Error Type":
                    continue
                exceptions.append(cells[1].rsplit(".", 1)[-1])
            documented_keys.update(exceptions)

except Exception as e:
    raise Exception(
        f"Error reading documentation: {e}, \n repo base - {os.listdir(repo_base)}"
    )

# Compare and find undocumented keys
undocumented_keys = error_names - documented_keys

if undocumented_keys:
    raise Exception(
        f"\nKeys not documented in 'LiteLLM Exceptions': {undocumented_keys}"
    )
