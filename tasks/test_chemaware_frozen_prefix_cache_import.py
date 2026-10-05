"""Prevent direct training from reintroducing legacy ChemAware probe imports."""
from __future__ import annotations

import inspect

from chemaware_frozen_prefix_cache import FrozenPrefixSpectrumStore


def main() -> None:
    source = inspect.getsource(FrozenPrefixSpectrumStore)
    assert "frozen_probe_v3" not in source
    assert "train_chemaware_shared_v3_peft" not in source
    assert "dependency_minimal_direct_cache_v1" in source
    print("dependency-minimal frozen-prefix cache import passed")


if __name__ == "__main__":
    main()
