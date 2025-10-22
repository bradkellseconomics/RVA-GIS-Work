import argparse
import os
import subprocess
import sys
from typing import List, Tuple


SCRIPTS = [
    ("geohub", "get_geohub_files.py"),
    ("servicelines", "serviceline_view_download.py"),
    ("race_ethnicity", "race_and_ethnicity.py"),
]


def run_script(label: str, path: str) -> Tuple[str, int]:
    print(f"\n=== [{label}] {path} ===")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return (label, 0)
    proc = subprocess.run([sys.executable, "-u", path])
    return (label, proc.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run all data download scripts in sequence.")
    ap.add_argument(
        "--only",
        choices=[k for k, _ in SCRIPTS],
        nargs="*",
        help="Run only selected components (default: all)",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any script fails (default: continue)",
    )
    args = ap.parse_args()

    # Ensure unified raw folder exists
    os.makedirs("data_raw", exist_ok=True)

    targets: List[Tuple[str, str]] = SCRIPTS
    if args.only:
        only_set = set(args.only)
        targets = [(k, p) for (k, p) in SCRIPTS if k in only_set]

    results = []
    for label, path in targets:
        lbl, code = run_script(label, path)
        results.append((lbl, code))
        if code != 0 and args.stop_on_error:
            print(f"[ERROR] {label} failed with code {code}; stopping early.")
            break

    print("\n=== Summary ===")
    for lbl, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"- {lbl}: {status}")

    # Non-zero exit if any failed
    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

