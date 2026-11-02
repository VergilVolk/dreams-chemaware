# DreaMS Unified Additive Full23 v1.3.1

This bundle is the incremental Full23 v1.3.1 consumed-development release. It reuses the Full23 v1 parent and v1.3 runtime, and replaces only the tensor-set readout with the selected `v13_01` checkpoint. The executable composition entrypoint is `runtime/full23_v131_release.py::FrozenFull23V131`.

The change fixes residual saturation by regularizing the query-centred raw residual and removes the historical twofold training multiplier. It was fit on consumed Enveda-180 meta-development and selected on 11,019 unique GNPS development queries. It is not untouched external confirmation.

The exact result and limitations are recorded in `docs/FULL23_V13_CUMULATIVE_REPAIRS_RESULT_20261102.md`.
