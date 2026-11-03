# CALMS architecture artwork provenance

The architecture artwork was produced with the built-in image-generation tool and refined through targeted visual edits. The final raster contains no generated prose; every reader-facing label is added deterministically by `tasks/render_calms_project_summary.py`.

## Final edit prompt

Preserve a wide scientific workflow with one LC-MS/MS input panel, three parallel evidence panels, one blank CALMS integration panel and one ranked annotation-output panel. Increase the architecture height by approximately 15--18% so the workflow no longer looks vertically compressed. Keep pale-academic-blue title headers on the input, evidence and output panels; keep the central CALMS panel header-free. Retain three distinct MS/MS spectra in the input and leave three blank structure slots below them. Retain the diagnostic peaks, fragment arrows and check/cross marks in the fragmentation panel, but leave all candidate-structure slots blank. Redesign the output as three substantial full-width candidate cards with rank badges, a blank central structure slot and a check on rank 1. Remove every generated molecular structure from the raster. Use a pure white background, very pale academic-blue fills, navy line art and thin blue arrows. Include no prose, pseudo-text, watermark, shadow or 3D effect.

All visible molecular structures are drawn after generation by RDKit from fixed SMILES, not by the image model. The three candidates are the valid C7H8O structural isomers anisole (`COc1ccccc1`), benzyl alcohol (`OCc1ccccc1`) and o-cresol (`Cc1ccccc1O`).

## Reader-facing labels added in code

- LC-MS/MS spectra and candidate structures
- Condition-robust spectral representation
- Fragmentation chemistry-guided discrimination
- Biochemical relationships in biological samples
- CALMS metabolite annotation
- Annotated candidate structures

## Palette

- Panel fill: `#F4F9FC` / `#EAF4FA`
- Secondary strokes: `#A9CDE2`
- Emphasis blue: `#4F91BC`
- Primary text and lines: `#214F73`
