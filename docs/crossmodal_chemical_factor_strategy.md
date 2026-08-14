# Cross-modal chemical-factor discovery: revised strategy

## Scientific question

Can a reproducible direction in frozen DreaMS space be assigned a chemical
meaning that is supported independently by molecular structure and by MS/MS
peak evidence?

A candidate factor is represented as a bundle rather than a single latent
coordinate:

\[
F_k=(u_k^{MS},v_k^{Mol},\rho_k,\ell_k^{descriptor},
     a_k^{atom/subgraph},e_k^{peak/loss}).
\]

It is accepted as a chemical factor only when it:

1. replicates on a molecule-disjoint confirmation set;
2. is stronger than an exact-mass-only baseline and is not explained by
   adduct, instrument, or collision energy;
3. is shared by at least two independent molecular views;
4. has an interpretable descriptor or atom/subgraph loading;
5. is supported by repeated peak or neutral-loss perturbation evidence from
   independent molecules.

## Molecular views and their roles

| View | Role in this project | Priority |
|---|---|---:|
| Selected RDKit/Mordred descriptors | Transparent naming and mass-confound control | P0 |
| MolFormer | SMILES-language global molecular semantics | P1 |
| KPGT/LiGhT | 2D topology, bonds, paths, descriptors and fingerprints | P1 |
| Uni-Mol | 3D/conformer-sensitive residual information | P2 |
| Chemprop/D-MPNN | Architecture or task-specific checkpoint, not a universal teacher by itself | P2 |
| ADMETlab/ADMET-AI | Endpoint-biased property validators, not primary factor teachers | P3 |

## Completed P0 descriptor pilot

- Frozen raw SSL and official fine-tuned DreaMS checkpoints were evaluated.
- Discovery: 464 molecules; confirmation: 464 molecules; IK14 overlap: 0.
- Forty transparent descriptors remained after excluding exact mass and
  zero-residual variables.
- A 12-knot cubic-spline exact-mass baseline was fitted on discovery only.
- Descriptor probes and PLS factors were fitted on discovery and evaluated on
  confirmation.

Preliminary results:

| Result | Raw SSL | Official fine-tuned |
|---|---:|---:|
| Residual descriptors with confirmation Spearman >= 0.3 | 28/40 | 28/40 |
| Residual descriptors with confirmation R2 > 0 | 37/40 | 37/40 |
| Independently refitted PLS axes with cosine >= 0.7 | 2/6 | 2/6 |

The two most reproducible candidate axes are:

1. aliphatic/saturated character versus aromaticity, dominated by
   FractionCSP3, aromatic ring count, Chi0v and surface/topology terms;
2. nitrogen-versus-oxygen composition, dominated by N/O counts and fractions.

These are candidate chemical coordinates, not yet mechanisms.  Their external
cross-view correlations are strong, but they still require stricter precursor
mass/adduct controls and peak-level validation.  The raw SSL representation is
at least as chemically decodable as the official fine-tuned representation on
this pilot, so official fine-tuning cannot be assumed to improve every type of
chemical information.

## Next experiment

1. Extract frozen MolFormer and KPGT/LiGhT embeddings for the same 928
   molecule-disjoint structures.
2. Compare shallow probes under the same split and exact-mass baseline.
3. Keep only molecular teachers that add confirmation-set information beyond
   the transparent descriptor panel.
4. Fit sparse multi-view CCA/PLS across DreaMS, MolFormer, KPGT and descriptors.
5. Map surviving axes to atoms/subgraphs and then to MS/MS peaks by occlusion.
6. Add or modify chemical rules only when an unrepresented peak/subgraph motif
   repeats across independent molecules.

This sequence changes the rule-library workflow from expert-first expansion to
evidence-driven completion.

## MolFormer experiment (completed)

### Extraction audit

- Official model: `ibm-research/MoLFormer-XL-both-10pct`, `compat-v4` revision.
- Frozen 768-dimensional embeddings were extracted for all 928 structures.
- Repeated inference maximum absolute error: 0.
- 274/928 inputs lost stereochemical notation under the model-compatible
  non-isomeric SMILES policy.
- One true stereochemical input collision (two records) occurred in the
  confirmation cohort and is not eligible for stereochemical claims.

### Teacher geometry

On the exact undirected 10 ppm neighbor graph, MolFormer cosine similarity was
strongly correlated with Morgan Tanimoto:

| Split | Neighbor links | Spearman rho |
|---|---:|---:|
| Discovery | 404 | 0.847 |
| Confirmation | 406 | 0.799 |

The confirmation median cosine was 0.935 for same-scaffold links and 0.755 for
different-scaffold links.  Therefore, the molecular teacher has meaningful 2D
structural geometry on this cohort; poor spectrum-to-molecule projection cannot
be attributed to a broken MolFormer representation.

### Frozen shallow alignment

A PCA plus ridge projection was trained on discovery DreaMS--MolFormer pairs and
evaluated on confirmation 10 ppm candidates:

| DreaMS checkpoint | Cross-modal AUC | Top-1 |
|---|---:|---:|
| Raw SSL | 0.561 | 0.518 |
| Official fine-tuned | 0.567 | 0.525 |

This low-data linear map is not a usable retrieval model.  Its purpose was to
test whether a frozen molecular teacher exposes stable chemical directions.

The leading shared factors had external correlations around 0.71--0.83 and were
again dominated by nitrogen/aromaticity and oxygen/polarity.  Strict independent
direction replication retained one factor for raw SSL and no factor with both
directions above 0.7 for the official checkpoint (one official factor had a
joint geometric-mean direction cosine above 0.7).

Transparent mass-residual descriptors explained 30.6% of external MolFormer
variance.  After removing this descriptor-explainable part, external shared
factor correlations fell to 0.10--0.29 and zero of six directions replicated.

Current decision: MolFormer is a valid external validator for broad 2D chemical
composition and topology, but it has not yet supplied a stable new factor beyond
the transparent descriptor panel.  It should not be used as a distance label or
as the sole teacher for DreaMS fine-tuning.

KPGT remains scientifically attractive as a second, bond-centered molecular
view.  Its official environment, however, is Linux/Python 3.7/PyTorch 1.10 with
DGL-CUDA 0.7.2, and should not be installed into the current Windows DreaMS
environment.  It requires an isolated environment or server-side extraction.
