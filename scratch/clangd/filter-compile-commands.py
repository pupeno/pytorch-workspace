#!/usr/bin/env python3
"""Shrink PyTorch's compile_commands.json so clangd's background index finishes.

The raw database cmake emits has 3808 translation units. Indexing all of them
takes clangd well over a day in this container, and ~half the entries produce
nothing useful:

  * third_party/*  - vendored deps. Their *headers* still get indexed
                     transitively via the PyTorch TUs that include them; only
                     their own .cpp implementations drop out.
  * nvcc entries   - clangd does not understand nvcc command lines. In a 5h run
                     it produced zero .cu index shards and parked two workers on
                     a generated kernel that does not exist on disk.
  * test/          - C++ gtest binaries, rarely navigated.

Re-run this after any cmake reconfigure, which rewrites the raw database.
A pristine copy of the raw database is kept next to this script.
"""

import argparse
import json
import os
import shutil
import sys

REPO = "/workspaces/pytorch/pytorch"
BUILD_CDB = os.path.join(REPO, "build", "compile_commands.json")
ROOT_CDB = os.path.join(REPO, "compile_commands.json")
RAW_COPY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile_commands.raw.json")


def relpath(entry):
    f = entry["file"]
    prefix = REPO + "/"
    return f[len(prefix):] if f.startswith(prefix) else f


def command_of(entry):
    if "command" in entry:
        return entry["command"]
    return " ".join(entry.get("arguments", []))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-third-party", action="store_true")
    p.add_argument("--keep-cuda", action="store_true")
    p.add_argument("--keep-tests", action="store_true")
    p.add_argument("--restore", action="store_true",
                   help="put the unfiltered database back and exit")
    args = p.parse_args()

    if args.restore:
        if not os.path.exists(RAW_COPY):
            sys.exit(f"no pristine copy at {RAW_COPY}")
        shutil.copyfile(RAW_COPY, BUILD_CDB)
        link_root()
        print(f"restored unfiltered database ({len(json.load(open(BUILD_CDB)))} entries)")
        return

    db = json.load(open(BUILD_CDB))

    # Self-healing: if cmake has rewritten the database, re-snapshot it.
    if is_unfiltered(db) or not os.path.exists(RAW_COPY):
        shutil.copyfile(BUILD_CDB, RAW_COPY)
        print(f"snapshotted raw database -> {RAW_COPY}")
    else:
        db = json.load(open(RAW_COPY))
        print(f"database already filtered; re-filtering from {RAW_COPY}")

    kept, dropped = [], {"third_party": 0, "cuda": 0, "tests": 0}
    for e in db:
        rel = relpath(e)
        if not args.keep_third_party and rel.startswith("third_party/"):
            dropped["third_party"] += 1
            continue
        if not args.keep_cuda and "nvcc" in command_of(e):
            dropped["cuda"] += 1
            continue
        if not args.keep_tests and rel.startswith(("test/", "c10/test/")):
            dropped["tests"] += 1
            continue
        kept.append(e)

    with open(BUILD_CDB, "w") as fh:
        json.dump(kept, fh, indent=1)
    link_root()

    total = len(db)
    print(f"{total} -> {len(kept)} entries ({100 * len(kept) / total:.0f}% kept)")
    for name, n in dropped.items():
        print(f"  dropped {n:5d}  {name}")


def is_unfiltered(db):
    return any(relpath(e).startswith("third_party/") for e in db)


def link_root():
    """Write the filtered database to the repo root as well.

    clangd loads a separate database per directory: files under build/ resolve to
    build/compile_commands.json, everything else to the repo-root copy. Both must
    carry the same filtered set or the build-side one re-queues everything.

    This is a real copy, not a symlink. PyTorch's own `merge_compile_commands`
    ALL target (cmake/PostBuildSteps.cmake) regenerates the repo-root file on
    every build via tools/merge_compile_commands.py, and its write_text() would
    follow a symlink and clobber the filtered build/ database through it.
    """
    if os.path.islink(ROOT_CDB):
        os.remove(ROOT_CDB)
    shutil.copyfile(BUILD_CDB, ROOT_CDB)


if __name__ == "__main__":
    main()
