# DreaMS-VOLAR v1.0

**DreaMS-VOLAR** expands to **Vector, Orthogonal and Local Evidence with an
Additive Residual for MS/MS Candidate Ranking**. `Full23 v1.3.2` is the frozen
implementation alias retained for exact provenance and compatibility.

DreaMS-VOLAR v1.0 is the executable candidate-ranking composition

```text
Full23 v1 score + 0.9 * (Full23 v1.3.1 score - Full23 v1 score)
```

It keeps the exact Full23 v1 scalar-evidence parent and the exact v1.3.1
four-encoder tensor-set readout. No network or upstream encoder was refit for
v1.3.2. The only change is a fixed residual-amplitude calibration.

The package entrypoint is `runtime/full23_v132_release.py`, class
`FrozenFull23V132`. It requires the same aligned candidate axis, scalar
evidence, four embedding banks, query/reference positions, reference masks and
embedding-bank provenance as v1.3.1. Missing representation evidence keeps the
valid Full23 v1 parent score.

On consumed GNPS development, the fixed scale changed v1.3.1 as follows:

- identity-disjoint: 10,995 queries, 88.3311% to 88.3674% Recall@1;
- formula-disjoint: 5,261 queries, 89.5647% to 89.6027% Recall@1;
- query-balanced 11,019-query endpoint: 88.3928% to 88.4245% Top-1.

The 0.9 value was identified after inspecting consumed GNPS predictions. This
is therefore a post-result development calibration, not an unopened external
confirmation. Enveda-180 is also consumed meta-development. A genuinely new
source is required for confirmation.

ChemAware V2 remains part of the Full23 scalar parent where candidate chemical
evidence is available. BioAware is not part of this closed-library ranker and
must be applied separately only when real sample-level event context exists.
