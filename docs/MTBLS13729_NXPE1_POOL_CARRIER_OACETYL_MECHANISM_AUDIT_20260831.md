# MTBLS13729 Neu5Ac pool—NXPE1—MUC2 O-acetylation mechanism audit (2026-08-31)

## Executive decision

The strongest evidence-supported model is **not** a single linear pathway from free Neu5Ac to an
O-acetylated end product. It is a three-layer, partially decoupled system:

1. **Pool:** Level-1 free Neu5Ac is expanded in all 10 Rmu tumour–normal pairs.
2. **Carrier programme:** NXPE1 is relatively enriched in mucinous versus conventional CRC, but this
   enrichment is absorbed by a secretory-mucin expression programme.
3. **Destination:** external MUC2 glycoproteomics shows tumour-associated truncation and loss or
   redistribution of O-acetylated sialic-acid glycoforms, whereas local bulk mono-O-acetyl-Neu5Ac-like
   exact-mass peaks do not increase with free Neu5Ac.

The resulting manuscript-level interpretation is:

> Mucinous CRC shows expansion of a free Neu5Ac pool embedded in a secretory-mucin state, but the
> activated donor and detectable bulk O-acetylated-sialic-acid-like products do not expand in parallel.
> This supports pool–carrier–destination decoupling rather than uniform hypersialylation or increased
> O-acetylation flux.

## 1. Local paired metabolite evidence

Source: 10 Rmu tumour–matched-normal pairs in the original HILIC(-) supplement.

| Node | Identity | Positive pairs | Mean paired log2 change | Bootstrap 95% CI |
|---|---|---:|---:|---:|
| free Neu5Ac | Level 1 | 10/10 | +2.249 | [1.641, 2.866] |
| CMP-Neu5Ac | Level 2 | 6/10 | +0.556 | [-0.157, 1.410] |
| UDP-GlcNAc | Level 1 | 5/10 | +0.327 | [-0.993, 1.687] |

The pre-specified within-patient contrasts are positive:

- free Neu5Ac minus CMP-Neu5Ac: `+1.693 log2`, bootstrap lower bound `+0.710`, Holm-Wilcoxon
  `p=0.0273`;
- free Neu5Ac minus UDP-GlcNAc: `+1.922 log2`, bootstrap lower bound `+0.527`, Holm-Wilcoxon
  `p=0.0273`.

This is direct evidence for a free-pool/activated-donor mismatch. It does not specify synthesis,
salvage, release, uptake, de-O-acetylation, reduced incorporation, subcellular localisation or flux.

## 2. NXPE1 current-GDC audit

### 2.1 Frozen cohort and primary endpoint

- primary cohort: the exact previously locked 371 TCGA COAD/READ primary tumours;
- histology: 42 mucinous and 329 conventional;
- primary endpoint: NXPE1 mucinous coefficient using current GDC STAR TPM after clinical and
  non-overlapping broad-lineage adjustment;
- sensitivity: MSI adjustment and STAR FPKM-UQ processing;
- secretory-mucin sensitivity programme: `MUC2`, `TFF3`, `SPDEF`, `FCGBP`, `AGR2`.

The extended current-GDC cohort is reported only as sensitivity and does not replace the locked cohort.

### 2.2 Results

| Model | TPM beta | p | FPKM-UQ beta | p |
|---|---:|---:|---:|---:|
| clinical | +0.563 | 0.00236 | +0.583 | 0.00182 |
| clinical + lineage | +0.621 | 0.000369 | +0.636 | 0.000279 |
| clinical + lineage + MSI | +0.530 | 0.00134 | +0.543 | 0.00130 |
| clinical + lineage + secretory mucin | +0.064 | 0.734 | +0.106 | 0.580 |
| clinical + lineage + MSI + secretory mucin | -0.048 | 0.782 | -0.013 | 0.942 |

Thus NXPE1 mucinous enrichment is real across two processing units, but it is **not independent of the
secretory-mucin state**. The correct interpretation is carrier-linked O-acetylation capacity, not an
independent NXPE1 driver.

This attenuation is not driven by one arbitrarily chosen marker. In both TPM and FPKM-UQ, every
leave-one-marker-out version of the five-gene secretory programme keeps the NXPE1 mucinous coefficient
non-significant. The TPM leave-one-out coefficients range from `-0.063` to `+0.033` and the FPKM-UQ
range is `-0.028` to `+0.065`; no two-marker adjustment retains significance in either unit. This
supports a distributed secretory carrier state, while remaining a covariate-sensitivity analysis rather
than causal mediation.

In 50 current-GDC tumour–normal pairs, NXPE1 is lower in tumour in 47/50 pairs (TPM mean tumour-minus-
normal `-2.709 log2`; exact sign `p=3.71e-11`). This is not contradictory: general CRC suppresses NXPE1
relative to normal colon, while the mucinous subset retains or re-expresses more NXPE1 than conventional
CRC because it retains a stronger secretory-mucin programme.

### 2.3 Independent mucinous single-cell context

The six paired mucinous CRC samples in GSE236696 provide an independent tumour-versus-adjacent-normal
composition sensitivity check. Using a conservative broad epithelial gate and patient-level pseudobulk,
NXPE1 decreases in all six tumour samples (mean tumour-minus-normal `-1.084 log2`; bootstrap 95% CI
`[-1.707, -0.484]`; exact two-sided sign-flip `p=0.0625`). NXPE1 is low-count, including one pair with
zero counts on both sides, so this is directional context rather than an independent significant
replication. It agrees with the current-GDC general-CRC paired result, but cannot test mucinous versus
conventional CRC because GSE236696 contains no conventional comparator.

The deposited GSE236696 feature index omits `MUC2` in all 12 samples. A secondary gate based on the four
available markers `TFF3/SPDEF/FCGBP/AGR2` yields only two complete patient pairs and is not interpretable.
The absent `MUC2` feature is therefore recorded as technical unavailability, never as zero expression.

A separate published single-cell comparison of 3 mucinous and 4 classical adenocarcinomas reports that
mucinous cancer cells are enriched for the goblet/secretory markers `MUC2`, `FCGBP`, `REG4` and `SPINK4`.
This is independent subtype context for the carrier programme, but its raw human data (`HRA003634`) are
controlled-access and the paper does not report NXPE1 as a validated endpoint. It is therefore not counted
as an NXPE1 replication.

### 2.4 Substrate caveat

NXPE1 must not be described as an exclusively free-Neu5Ac enzyme. Two 2025 primary studies use different
acceptor contexts:

- the JACS study reports regioselective 9-O-acetylation of Neu5Ac and mucin-glycan modification;
- the Nature Communications study demonstrates NXPE1-dependent acetylation with CMP-Neu5Ac in vitro.

Therefore local free Neu5Ac is a measured pool node, not a proven direct in-vivo NXPE1 substrate.

## 3. Product-side evidence and counter-evidence

### 3.1 Original supplement

A full read-only audit of all five XLSX and three CSV supplements found Level-1 Neu5Ac and Level-2
CMP-Neu5Ac but no named mono/di/tri-O-acetyl-Neu5Ac row and no numeric match to the corresponding
negative-ion exact masses within 10 ppm. This prevents retrospective relabelling of an author-reported
metabolite as the product.

### 3.2 Local negative-HILIC raw data

Phenotype-blind extraction at `m/z 350.109269 [M-H]-` found two reproducible RT peaks:

- 4.29 min: 50/60 samples, 47 RT-resolved MS2;
- 5.55 min: 54/60 samples, 56 RT-resolved MS2.

Both contain a strong `m/z 87.0088` motif but cannot distinguish 4/7/8/9-O-acetyl positional isomers.
Neither peak increases reproducibly in Rmu (both BH `q=0.930`), and patient-level changes do not correlate
with Level-1 free Neu5Ac (`rho=0.170` and `-0.067`). These are exact-mass family signals and a useful
negative result, not identified products.

### 3.3 External MUC2 glycoproteomics

PXD055865 contains one healthy colon donor and three colorectal tumour specimens representing only two
independent tumour patients. It provides structural/existence evidence that MUC2 carries mono-, di- and
tri-O-acetylated Neu5Ac glycoforms and that colorectal tumours contain less of these forms than the healthy
colon, together with abundant truncated T/Tn-rich MUC2 glycoforms. It does not provide a population-level
abundance replication and does not measure free Neu5Ac.

This external result supplies the missing **carrier/destination context** while retaining a strict sample-size
boundary.

## 4. Updated mechanism model

The evidence supports a hybrid mucin-glycome model with four separable coordinates:

1. **donor/pool:** free Neu5Ac accumulation;
2. **activated donor/transport:** no matched expansion of CMP-Neu5Ac despite a mucinous-relative
   synthesis/transport transcript background;
3. **carrier:** secretory MUC2/goblet-like programme, within which NXPE1 expression is embedded;
4. **core/linkage/final modification:** core-3/Sda retention, core-2/sLeX acquisition, alpha2-6 loss and
   heterogeneous O-acetyl destinations.

The decisive innovation is the **decoupling**, not a claim of global hypersialylation.

## 5. What is established and what remains open

### Established within current evidence

- robust same-patient expansion of Level-1 free Neu5Ac in Rmu;
- direct free-pool versus activated-donor mismatch;
- reproducible mucinous-relative NXPE1 enrichment before secretory-mucin adjustment;
- NXPE1 enrichment is carrier-program-linked rather than independent;
- independent six-pair mucinous single-cell data directionally agree that tumour epithelium does not
  show a general NXPE1 increase over adjacent normal;
- bulk mono-O-acetyl-Neu5Ac-like exact-mass peaks do not track free Neu5Ac;
- external MUC2 glycoproteomics supports tumour-associated destination remodelling.

### Not established

- NXPE1 protein abundance or enzymatic activity in the MTBLS13729 samples;
- whether free Neu5Ac is directly consumed by NXPE1 in vivo;
- the identity or position of either local `m/z 350.109269` feature;
- same-sample MUC2 glycoform abundance;
- isotope flux, enzyme causality, microbial contribution or treatment relevance;
- independent patient-level mucinous metabolite replication.

## 6. Highest-value next validation

1. **Minimum identity upgrade:** Neu5Ac authentic standard, same-method RT/MS2, pooled-extract spike-in
   and ideally an isotope-labelled internal standard.
2. **Destination upgrade:** linkage-aware O-glycomics or MUC2 glycopeptide analysis in a larger public or
   collaborator cohort; this is more informative than more bulk transcript scores.
3. **If pursuing O-acetyl position:** paired 4-O- and 9-O-acetyl-Neu5Ac standards; if ordinary LC-MS cannot
   resolve them, use IM-MS/CCS rather than over-interpreting common fragments.
4. **Minimal tissue validation if material ever becomes available:** NXPE1 protein plus an O-acetyl-sensitive
   histochemical or lectin readout co-localised with MUC2. This tests the carrier-linked model but still does
   not establish flux.

## 7. Reproducible artefacts

- `data/mtbls13729/sialic_donor_decoupling_v1/report.json`
- `data/mtbls13729/oacetyl_neu5ac_like_v2/report.json`
- `data/mtbls13729/source_paper_supplements/oacetyl_sialic_audit_v1.json`
- `data/external/TCGA_COADREAD_Xena_20260830/nxpe1_free_donor_v3_secretory/report.json`
- `data/external/TCGA_COADREAD_Xena_20260830/nxpe1_secretory_sensitivity_v1/report.json`
- `data/external/GSE236696/nxpe1_secretory_epithelial_v2/report.json`
- `data/external/HRA003634_MC_vs_AC_supplement/access_audit.json`
- `data/external/PXD055865_2026_MUC2/source_data_audit_v1/`
- `tasks/audit_tcga_nxpe1_free_donor_mechanism_v1.py`
- `tasks/validate_tcga_nxpe1_free_donor_mechanism_v1.py`
- `tasks/audit_tcga_nxpe1_secretory_program_sensitivity_v1.py`
- `tasks/audit_gse236696_nxpe1_secretory_epithelial_v1.py`
- `tasks/audit_mtbls13729_oacetyl_neu5ac_like.py`
- `tasks/audit_mtbls13729_supplement_oacetyl_sialic.mjs`

## 8. Claim boundary

This is an evidence-calibrated multi-dataset mechanism model. It supports a static abundance and carrier-
state interpretation, not metabolic flux, enzyme activity, molecular causality or a clinically validated
biomarker.
