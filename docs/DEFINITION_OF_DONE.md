# Definition of Done

## Core P0 (must pass before any extension is promoted)
- pytest: 0 failures
- document indexing success ratio >= release policy
- vector/metadata ID consistency
- holdout evaluation thresholds pass
- citation_success satisfies release policy
- reproducibility manifest exists
- strict mode: Python version, lock file, pip check pass

## Extension promotion
An extension is candidate-only until:
1. core tests still pass,
2. candidate metrics are written to JSON,
3. `scripts/promote_feature.py FEATURE candidate.json` returns 0,
4. feature flag is enabled only after the successful promotion,
5. rollback path is documented and tested.

P1/P2 failure never changes the last known-good Core baseline.
