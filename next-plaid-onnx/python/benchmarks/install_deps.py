"""Install pylate for the export benchmark, working around missing wheels.

`fast-plaid` (a pylate dependency) publishes no linux-aarch64 wheels, so a plain
`pip install pylate` cannot resolve on ARM Linux -- and unpinned, pip silently
backtracks to pylate 1.2.0, whose ColBERT predates the `do_query_expansion`
kwarg the exporter passes.

Export only uses `pylate.models.ColBERT`, which imports numpy, torch, scipy and
sentence_transformers and never touches fast-plaid (verified by blocking the
import). So on platforms where the normal install fails, install pylate with
--no-deps and then install its OWN declared requirements minus fast-plaid.

Deriving the requirement list from metadata matters: pylate pins
`sentence-transformers==5.3.0` exactly, and installing a newer one produces
`KeyError: 'activation_function'` when loading the Dense module.

  python install_deps.py [--pylate ">=1.6.0"]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

SKIP = {"fast-plaid", "fast_plaid"}


def pip(*args: str) -> int:
    print(f"$ pip {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, "-m", "pip", "install", *args])


def pylate_requirements() -> list[str]:
    import importlib.metadata as md

    out = []
    for raw in md.requires("pylate") or []:
        if "extra ==" in raw:  # optional groups
            continue
        req = raw.split(";")[0].strip()
        name = req.split("[")[0]
        for ch in ("=", "<", ">", "!", "~", " "):
            name = name.split(ch)[0]
        if name.lower().replace("_", "-") in SKIP:
            print(f"  skipping {req} (no wheel on this platform)", flush=True)
            continue
        out.append(req)
    return out


def verify() -> None:
    import inspect

    from pylate import models

    params = inspect.signature(models.ColBERT.__init__).parameters
    if "do_query_expansion" not in params:
        raise SystemExit("pylate too old: ColBERT has no `do_query_expansion`")
    print("pylate import OK", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylate", default=">=1.6.0")
    args = ap.parse_args()
    spec = f"pylate{args.pylate}"

    if pip(spec) == 0:
        verify()
        return 0

    print(f"::warning::`pip install {spec}` failed here; retrying without fast-plaid",
          flush=True)
    if pip("--no-deps", spec) != 0:
        raise SystemExit(f"could not install {spec} even with --no-deps")

    reqs = pylate_requirements()
    if not reqs:
        raise SystemExit("pylate metadata listed no requirements; refusing to guess")
    if pip(*reqs) != 0:
        raise SystemExit("failed installing pylate's requirements")

    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
