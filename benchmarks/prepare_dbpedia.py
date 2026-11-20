"""DBpedia OpenAI 1M as ann-benchmarks builds it (dbpedia-openai-1000k-angular): the 1,000,000 OpenAI
text-embedding-ada-002 vectors (1,536 dimensions) of KShivendu/dbpedia-entities-openai-1M on Hugging
Face (MIT), split by scikit-learn's train_test_split(test_size=10_000, random_state=42) into base
vectors and queries, written to data/dbpedia-openai-1000k-angular.hdf5 (train, test). Reads the
dataset's Parquet files from data/dbpedia-openai/.

    uv run --group bench --with pyarrow --with scikit-learn python prepare_dbpedia.py
"""
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    files = sorted((DATA / "dbpedia-openai").glob("*.parquet"))
    parts = []
    for f in files:
        col = pq.read_table(f, columns=["openai"]).column("openai").combine_chunks()
        parts.append(np.asarray(col.flatten(), dtype=np.float32).reshape(-1, 1536))
    x = np.concatenate(parts)
    print(f"{len(files)} files, {x.shape[0]:,} vectors of {x.shape[1]} dimensions")
    lengths = np.linalg.norm(x, axis=1)
    print(f"lengths as published: {lengths.min():.7f} to {lengths.max():.7f}")
    train, test = train_test_split(x, test_size=10_000, random_state=42)
    out = DATA / "dbpedia-openai-1000k-angular.hdf5"
    with h5py.File(out, "w") as h:
        h.attrs["distance"] = "angular"
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
    print(f"wrote {out.name}: train {train.shape}, test {test.shape}")


if __name__ == "__main__":
    main()
