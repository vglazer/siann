#!/bin/sh
# Copies the documentation pages, which live at the top of the repository (where GitHub shows them),
# into docs/src for MkDocs, which cannot build from the folder holding its config. Run before
# `mkdocs build`; Read the Docs runs it too (.readthedocs.yaml).
set -e
cd "$(dirname "$0")/.."
rm -rf docs/src
mkdir -p docs/src
cp README.md ARCHITECTURE.md ALGORITHMS.md BACKENDS.md DATASETS.md BENCHMARKS.md TUNING.md docs/src/
# Links to the benchmark folders point at GitHub, which shows them; the docs site has only the pages.
perl -pi -e 's#\]\(benchmarks/(results/)?\)#](https://github.com/vglazer/siann/tree/main/benchmarks/$1)#g' docs/src/*.md
perl -pi -e 's#\]\(examples/([^)]*)\)#](https://github.com/vglazer/siann/blob/main/examples/$1)#g' docs/src/*.md
