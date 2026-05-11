"""
get_versions.py
---------------
Detects installed versions of specified packages and outputs a conda
environment YAML (environment.yml) ready for `conda env create -f environment.yml`.

Usage:
    python get_versions.py
    python get_versions.py --name my_env
    python get_versions.py --output my_requirements.yml
"""

import importlib
import argparse
import sys
from pathlib import Path

# Packages to check.
# Format: (import_name, conda/pip_package_name)
PACKAGES = [
    ("pyro",        "pyro-ppl"),
    ("torch",       "pytorch"),
    ("tqdm",        "tqdm"),
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("polars",      "polars"),
    ("gpytorch",    "gpytorch"),
    ("pyproj",      "pyproj"),
    ("rasterio",    "rasterio"),
]

# Packages that are conda-forge only (not in defaults channel)
CONDA_FORGE = {"pyro-ppl", "gpytorch", "polars", "rasterio", "pyproj"}

# Packages that should be installed via pip inside conda
PIP_ONLY = set()


def strip_local(version: str) -> str:
    """Strip PEP 440 local version identifiers (e.g. +abc123, +cu118)."""
    import re
    return re.sub(r'\+.*$', '', version)


def get_version(import_name: str) -> str | None:
    """Return the version string for a package, or None if unavailable."""
    try:
        mod = importlib.import_module(import_name)
        for attr in ("__version__", "version", "VERSION"):
            v = getattr(mod, attr, None)
            if v and isinstance(v, str):
                return strip_local(v)
        # Fall back to importlib.metadata
        import importlib.metadata as meta
        return strip_local(meta.version(import_name))
    except Exception:
        return None


def build_env_yaml(env_name: str, results: list[dict]) -> str:
    lines = [
        f"name: {env_name}",
        "channels:",
        "  - conda-forge",
        "  - pytorch",
        "  - defaults",
        "dependencies:",
        f"  - python={sys.version_info.major}.{sys.version_info.minor}",
        "  - pip",
    ]

    pip_deps = []

    for r in results:
        pkg = r["pip_name"]
        ver = r["version"]
        spec = f"{pkg}=={ver}" if ver else pkg

        if pkg in PIP_ONLY:
            pip_deps.append(f"    - {spec}")
        else:
            lines.append(f"  - {spec}")

    if pip_deps:
        lines.append("  - pip:")
        lines.extend(pip_deps)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conda environment.yml from installed packages.")
    parser.add_argument("--name",   default="my_env",         help="Conda environment name (default: my_env)")
    parser.add_argument("--output", default="environment.yml", help="Output file path (default: environment.yml)")
    args = parser.parse_args()

    print(f"{'Package':<20} {'Import name':<15} {'Version':<15} {'Status'}")
    print("-" * 65)

    results = []
    for import_name, pip_name in PACKAGES:
        version = get_version(import_name)
        status = "found" if version else "NOT FOUND"

        print(f"{pip_name:<20} {import_name:<15} {version or '—':<15} {status}")
        results.append({"import_name": import_name, "pip_name": pip_name,
                         "version": version})

    yaml_content = build_env_yaml(args.name, results)
    output_path = Path(args.output)
    output_path.write_text(yaml_content)

    print(f"\nenvironment.yml written to: {output_path.resolve()}")
    print("\n--- environment.yml preview ---")
    print(yaml_content)
    print("--- end ---")
    print(f"\nTo create the environment, run:")
    print(f"  conda env create -f {output_path}")


if __name__ == "__main__":
    main()