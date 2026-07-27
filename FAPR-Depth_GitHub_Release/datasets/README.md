# Dataset preparation

This release does not redistribute TransCG, ClearGrasp, or ClearPose.

## Cache layout

The public training/evaluation scripts consume cached PyTorch shards:

```text
data/cache/
├── train/
│   ├── manifest.json
│   └── *.pt
├── val/
│   ├── manifest.json
│   └── *.pt
└── test/
    ├── manifest.json
    └── *.pt
```

Each `.pt` shard must contain:

- `rgb`
- `raw_depth`
- `gt_depth`
- `mask`
- `valid`
- `rel_aligned`

Optional fields used by the final model include `rel_conf`, `raw_prior`,
`rel_bg_resid`, `rel_bg_coverage`, `boundary`, and `base_final`.

A manifest has the form:

```json
{"shards": [{"file": "000001.pt"}, {"file": "000002.pt"}]}
```

Dataset licenses and original citations remain the responsibility of the user.
