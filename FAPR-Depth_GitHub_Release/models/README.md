# Model implementation

For checkpoint-key compatibility, the verified final model classes remain in
the root `train.py` research script. The principal class is:

```python
FailureAwarePosteriorDepth
```

Its modules include:

- metric alignment and multi-scale prior adaptation;
- four-state failure posterior estimation;
- missing, biased, and boundary experts;
- uncertainty-aware source allocation;
- safe-anchor residual fusion;
- candidate refinement and acceptance control.

Before a public release, keep the class/module names stable if existing
checkpoints will be distributed.
