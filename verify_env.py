"""Environment verification for EquiDrug.

Imports every library in the free stack and prints its version, confirming
the dependencies in requirements.txt install cleanly. Run this after
`pip install -r requirements.txt` (locally on CPU or on Colab).

Usage:
    python verify_env.py

Exit code is 0 if every library imports, 1 if any are missing.
"""

from __future__ import annotations

import importlib


# (pip package name, import module name, attribute holding the version)
LIBRARIES = [
    ("torch", "torch", "__version__"),
    ("torch-geometric", "torch_geometric", "__version__"),
    ("e3nn", "e3nn", "__version__"),
    ("rdkit", "rdkit", "__version__"),
    ("numpy", "numpy", "__version__"),
    ("pandas", "pandas", "__version__"),
    ("fastapi", "fastapi", "__version__"),
    ("uvicorn", "uvicorn", "__version__"),
]


def main() -> int:
    print("EquiDrug environment check")
    print("=" * 40)

    missing: list[str] = []
    for pip_name, module_name, version_attr in LIBRARIES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, version_attr, "unknown")
            print(f"  [ok]   {pip_name:18} {version}")
        except ImportError as exc:
            print(f"  [MISS] {pip_name:18} -> {exc}")
            missing.append(pip_name)

    print("=" * 40)

    # Report compute device via our central config (also exercises the import).
    try:
        from src.train.config import get_device

        print(f"Device detected: {get_device()}")
    except Exception as exc:  # noqa: BLE001 - report any config import issue
        print(f"Device detect:   could not import config ({exc})")

    if missing:
        print(f"\nFAILED: {len(missing)} package(s) missing: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return 1

    print("\nAll libraries imported successfully. Free stack is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
