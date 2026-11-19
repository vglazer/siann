#!/bin/sh
# Copies the documentation pages, which live at the top of the repository (where GitHub shows them),
# into docs/src for MkDocs, which cannot build from the folder holding its config. Run before
# `mkdocs build`; Read the Docs runs it too (.readthedocs.yaml).
set -e
cd "$(dirname "$0")/.."
rm -rf docs/src
mkdir -p docs/src/proto
cp README.md ARCHITECTURE.md ALGORITHMS.md BACKENDS.md DATASETS.md BENCHMARKS.md docs/src/
cp proto/RESULTS.md docs/src/proto/
# Links to the prototype folder point at GitHub, which shows it; the docs site has only the pages.
perl -pi -e 's#\]\(proto/\)#](https://github.com/vglazer/siann/tree/main/proto)#g' docs/src/*.md
