import sys
import filecmp
import shutil
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check or sync the canonical model cost map and bundled fallback mirror."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Copy the canonical root model map into the bundled fallback mirror when they differ.",
    )
    args = parser.parse_args(argv)

    print(
        "Comparing canonical model_prices_and_context_window.json and bundled fallback "
        "litellm/bundled_model_prices_and_context_window_fallback.json."
    )

    file1 = "model_prices_and_context_window.json"
    file2 = "litellm/bundled_model_prices_and_context_window_fallback.json"

    cmp_result = filecmp.cmp(file1, file2, shallow=False)

    if cmp_result:
        print(f"Passed! Files {file1} and {file2} match.")
        return 0
    else:
        print(
            f"Mismatch! Files {file1} and {file2} do not match."
        )
        if args.write:
            print(f"Copying content from {file1} to {file2}.")
            copy_content(file1, file2)
            return 0
        return 1


def copy_content(source, destination):
    shutil.copy2(source, destination)


if __name__ == "__main__":
    sys.exit(main())
