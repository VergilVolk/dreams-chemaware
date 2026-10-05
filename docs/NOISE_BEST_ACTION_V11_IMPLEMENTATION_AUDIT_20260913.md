# Noise V11 historical-best action integration audit

Date: 2026-09-13  
Status: local implementation and integration preflight passed; server result pending

## Implemented boundary

V11 is a two-module composition with a hard interface:

1. `HistoricalBestActionBankV1` selects the current-E8 strict Top-1,
   maximum-margin champion for each query from the seven mature action sources.
2. `ActionInjectorV1` consumes the resulting corrective gradient at the
   optimizer boundary without knowing the action source, recipe or spectrum.

The registered V11 and V10 configurations were compared field by field.  The
only exact differences are `direct_contract` and `action_bank_contract`; the
float configuration is identical.  The Injector V1 file was not edited and is
byte-locked by its integration test.

## Frozen digests

| artifact | SHA-256 |
|---|---|
| Injector V1 | `21e881f00942a61c2a192a4358806379215d55026f87c934c04f975da1887931` |
| historical-best supplier | `206e555d5eee706232a2d57151528dbb22eb4856685b3e853cb721d9490b6ae9` |
| action-bank compute-node audit | `0723df248b30991536f7aa2af422834def66b1584b19d1f13dec773a8db43c47` |
| integrated trainer | `1a3f80abcf7d1d7583d447bc0c315982760910113a0f3f375ecabeba234f1c5a` |
| V11 summarizer | `2653c94d44303799259a5c09468f5c06437a0a673e00fd7037ea9b113f151cc6` |
| V11 SBATCH | `83293d9d495c1617b716ba2dcc2a0f7988c34dce937f2701988335587c9eaf43` |
| 85-file source manifest | `71e6cf39b3571ae72f6a90a3387bf75e2452adc092e935af9e5ba19dbe9be4dd` |

## Local tests

The exact test sequence embedded in the Slurm job was executed locally with
the dependency-complete Python 3.13 runtime.  All 119 assertions passed:

| suite | tests |
|---|---:|
| Injector V1 golden equivalence | 7 |
| HistoricalBestActionBank V1 selection/integration | 5 |
| V11 summary transport and contract | 2 |
| V10 exact optimizer arbitration | 5 |
| V10 trainer contract | 5 |
| V6/V7 action admission regression | 11 |
| direct-v3 objective core | 22 |
| direct-v3 trainer helpers | 26 |
| V8 gradient locality | 4 |
| V4 optimizer arbitration | 16 |
| full-graph evaluation | 4 |
| V10 Slurm regression | 6 |
| V11 Slurm/resource/atomicity | 6 |

Both V10 and V11 SBATCH files also pass `bash -n`.  Both source manifests were
re-hashed against the current workspace.  V10 remains executable after the
trainer gained the new optional supplier because its default action-bank
contract remains `all_strict_top1` and the new import was added to its source
closure.

## Compute-node fail-closed gates

Before model construction V11 will:

- verify every one of the 85 source files and the manifest's own digest;
- compile and execute the complete test bundle from the immutable snapshot;
- require the registered ledger counts 32,127 / 3,483 / 3,482;
- require every one of the seven mature sources among selected champions;
- verify all frozen model/data/graph/ledger inputs;
- require two allocated GPUs;
- reject an existing run or final directory;
- require at least 6 GiB reported filesystem headroom;
- use no manual Slurm memory request.

Both arms in each wave are background workers governed by the same fail-fast
wait function.  The second-wave clean arm cannot replace the shell, so the V10
resume control-flow failure cannot recur.  The summary and final atomic move
run only after all four arms finish successfully.

## Scientific limit

The local result proves exact action selection, tensor preservation, source
closure, unchanged injection semantics and report transport.  It does not
prove a 4--5 pp shared-encoder gain.  The 5.333456 pp value is action-space
headroom; V11's formula-disjoint and outer-held results decide whether that
headroom transferred.
