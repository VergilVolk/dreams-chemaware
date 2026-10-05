"""Small parsing checks for corrected N routing."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_noise_corrected_n_router import (
    REGISTERED_FORMAL_N_ROUTE_CONFIGURATION,
    _validate_registered_formal_configuration,
    parse_control_paths,
)


def main() -> None:
    assert parse_control_paths("") == ()
    assert parse_control_paths(np.nan) == ()
    assert parse_control_paths("1,2;3,4") == ((1, 2), (3, 4))
    formal = dict(REGISTERED_FORMAL_N_ROUTE_CONFIGURATION)
    _validate_registered_formal_configuration(
        SimpleNamespace(**formal), formal=True,
    )
    formal["maximum_corrective_frontier"] = 15
    try:
        _validate_registered_formal_configuration(
            SimpleNamespace(**formal), formal=True,
        )
    except RuntimeError as error:
        assert "maximum_corrective_frontier" in str(error)
    else:
        raise AssertionError("formal N route accepted configuration drift")
    print("[noise corrected N router tests] PASS=4")


if __name__ == "__main__":
    main()
