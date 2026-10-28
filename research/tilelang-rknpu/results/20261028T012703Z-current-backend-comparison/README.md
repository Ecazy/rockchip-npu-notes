# Current backend comparison checkpoint

Runtime source: `74f6266`. Main report:
[CURRENT-BACKEND-COMPARISON.md](../../CURRENT-BACKEND-COMPARISON.md).

Five profiles, 10,000 MNIST test images per profile, three rounds. Each raw
sample records the same batch executed by current, RKNN and CPU arms in
rotating order. Current uses `stats=None`. Setup, warmup, argmax, reference
calculations and artifact writes are outside application timing. The main
latency is the median of the three primary-batch round means; p50 and tail
statistics are separate. Numerical policies remain distinct.

- `board/full.json`: schedules, round summaries, metrics, loaded runtime
  identity and array hashes. `board/smoke.json` is the coexistence check.
- `board/full_artifacts/raw_samples.jsonl.gz`: lossless compressed copy of
  all 68,442 batch records (each contains three arm durations).
- `board/full_artifacts/logits.npz` and `logits-index.json`: all 55 output and
  reference arrays, stored as nine unique arrays without changing any dtype,
  shape or element bytes. The index maps original filenames to archive keys
  and records original file and element hashes.
- `board/*.stdout`, `*.stderr`, `*.exit` and `*.runtime-mount.txt`: command
  outcomes and proof that the system RKNN runtime was restored after use.
- `campaign/run-manifest.json`: source commit and frozen package identities.
- `commands.txt`: deployment and measurement commands. Repeat from the
  repository root with
  `bash research/tilelang-rknpu/scripts/run-current-backend-comparison-host.sh`.

The original uncompressed arrays and JSONL remain available locally. They can
also be reconstructed from the committed archives:

```python
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

root = Path("board/full_artifacts")  # run from this checkpoint directory
with gzip.open(root / "raw_samples.jsonl.gz", "rb") as stream:
    (root / "raw_samples.jsonl").write_bytes(stream.read())
index = json.loads((root / "logits-index.json").read_text())
with np.load(root / index["archive"], allow_pickle=False) as arrays:
    for name, entry in index["members"].items():
        np.save(root / name, arrays[entry["key"]], allow_pickle=False)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry["npy_sha256"]
```

Model weights, dataset files, RKNN binaries, shared libraries and source bundles
remain local; their identities are recorded in the report and checksum lists.
`SHA256SUMS` inventories the complete local evidence tree, excluding itself.
