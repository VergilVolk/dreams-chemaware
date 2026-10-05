# Reverse standard-probe biology: audit, withdrawal, and corrected entry point

## 1. What is withdrawn

The earlier A0–A2 public-context classifier is not the reverse-metabolomics
biology module requested by the project.  It starts from public matches and
predicts metadata classes; it does not inject frozen reference spectra into
the 119 real MTBLS13729 RPLC sample runs.  It is retained only as an exploratory
metadata audit and must not be presented as the main biological innovation.

The first reverse-probe manifest (`reverse_probe_manifest_v1`) is also rejected
as a primary panel.  Its substring rules admitted combinatorial synthesis
entries such as reagent1_reagent2_Acetyl-L-Carnitine and inflated the panels to
119 carnitine, 1,345 nucleoside, 352 acetylated-polyamine and 65 sialic-acid
identities.  These are not clean endogenous-parent panels.

## 2. Correct task definition

The tensor is:

`frozen reference spectrum × real biological sample × evidence channel`.

The first evidence channels are precursor-constrained raw MS/MS, official
DreaMS embedding similarity, experimental shared-embedding similarity and a
bridge to the frozen MS1 feature/EIC matrix.  Phenotype is introduced only
after detection and calibration are frozen.

The scientific question is not whether similarity is mathematically symmetric.
The asymmetry is experimental: chemistry is selected first, then every sample
is queried for that chemistry, and only afterward are tissue and histology
distributions tested.

## 3. Frozen panel hierarchy

1. Source-paper Level-1 identities: positive calibration, not new discoveries.
2. Pre-existing MTBLS13729 hypotheses: palmitoylcarnitine, C20:4-acylcarnitine-like,
   N1,N8-diacetylspermidine-like, methyl/dimethyl-guanosine families and free Neu5Ac.
3. Strict clean-parent expansion: only uncombined parent spectra; combination
   products and known-isomer mixture names are excluded from the primary panel.
4. Synthetic derivative discovery remains a future, separately labelled panel.

The methylguanosine features remain formula/positional-isomer hypotheses.  A
Nelarabine or Guanosine_Acetaldehyde library record cannot upgrade them to an
exact endogenous identity.

## 4. Local audit result

The v4 manifest contains 153 spectra, 73 panel-specific identities and all 119
RPLC sample runs.  It includes 57/67 source Level-1 identities that have a public
reference spectrum and 17 panel-specific clean/frozen hypothesis identities.

The raw reverse scan produced 3,285 precursor-matched identity×sample pairs.
Among 2,707 source-Level-1 calibration pairs with an MS1 feature link, 1,971
selected the expected feature (72.81%).  This is evidence that the reverse
sample bridge is functioning; it is not a false-discovery estimate.

Raw MS/MS similarity alone is weak for choosing the correct co-mass feature:
sqrt-cosine AUC is approximately 0.57 on this same-cohort calibration.  This
pre-registers the actual role of DreaMS/E6: select the correct chromatographic
feature among precursor-compatible sample spectra, under the same candidate
graph, rather than merely report more library matches.

Several frozen hypotheses already show coherent raw bridges: palmitoylcarnitine
58/58 sample pairs, N1,N8-diacetylspermidine 46/46, the dimethylguanosine-family
probe 32/34 and positive-mode Neu5Ac 33/33 select their frozen feature.  C20:4
acylcarnitine-like is only 20/55, and the Nelarabine surrogate for the
methylguanosine family is poor.  The latter two are ambiguity diagnostics, not
negative biological conclusions.

## 5. Next frozen decision

Encode the exact 153 reference spectra with official DreaMS and the current
experimental shared embedding; search both into the same 119 sample embedding
manifests; compare expected-feature recovery on source Level-1 anchors.  Only a
nonnegative, reproducible calibration result permits phenotype analysis.

If that gate passes, phenotype analysis must use MS1 EIC abundance and paired
patient contrasts.  MS2 occurrence alone cannot be interpreted as absence or
quantitative abundance because DDA sampling is stochastic.

## Claim boundary

Public library reference spectra are not same-platform authentic standards.
No result here establishes MSI Level 1, enzyme activity, metabolic flux, or a
new exact metabolite identity.
