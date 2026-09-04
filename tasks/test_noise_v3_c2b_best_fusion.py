from __future__ import annotations
import numpy as np
from audit_noise_v3_c2b_best_fusion import minmax

def main():
    assert np.allclose(minmax(np.asarray([2., 4., 6.])), [0., .5, 1.])
    assert np.allclose(minmax(np.asarray([3., 3.])), [0., 0.])
    print("[test_noise_v3_c2b_best_fusion] PASS")
if __name__ == "__main__": main()
