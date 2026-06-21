# Reproducibility Notes

The released scripts provide a complete training and inference path for local
volumetric `.npy` data. Exact numerical comparison across institutions can be
affected by scanner metadata, preprocessing, normalization, patient splits,
hardware, and software versions.

Recommended practice:

1. Keep train/validation splits fixed in a manifest outside the repository.
2. Normalize each dataset consistently before saving `.npy` volumes.
3. Record the config YAML, random seed, checkpoint, and commit hash for each run.
4. Evaluate all compared methods on the same test volumes and slice range.

The synthetic demo is intended for installation and pipeline verification. It
is not a medical benchmark.
