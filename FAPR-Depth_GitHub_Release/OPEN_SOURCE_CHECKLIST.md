# Open-source release checklist

- [ ] Replace `USERNAME` in README and CITATION.cff.
- [ ] Confirm the software license with all co-authors.
- [ ] Remove private paths, emails, tokens, and unpublished data.
- [ ] Add the compatible third-party backbone under its license or document installation.
- [ ] Upload final checkpoints and publish SHA-256 checksums.
- [ ] Add a small legally redistributable example shard.
- [ ] Run `python -m compileall .`.
- [ ] Run training/evaluation smoke tests in a clean environment.
- [ ] Verify that reported metrics correspond to the released checkpoint.
- [ ] Confirm dataset terms permit the displayed qualitative examples.
- [ ] Add the paper DOI only after it is assigned.
