"""RAGAS is deliberately optional. Deterministic core metrics are the release gate.
RAGAS API changes must never block the core release. Adapt this module after a 2-case smoke test.
"""
def require_ragas():
    try:
        import ragas
        return ragas
    except ImportError as e:
        raise RuntimeError("RAGAS extra not installed; core evaluation is still available") from e
