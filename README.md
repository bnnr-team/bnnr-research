# bnnr-research

Research papers on the methods underlying BNNR.

## Papers

### `paper/` — Intelligent Coarse Dropout and Anti-ICD

A method paper introducing **ICD** (Intelligent Coarse Dropout) and **AICD**
(Anti-ICD): two saliency-guided masking augmentations for visual classifiers.
The paper presents the methods, the tile-based mask construction, the fill
strategies, and the reference implementation. It makes no empirical claims;
a quantitative evaluation is left to a companion study.

**Build the PDF:**

```bash
bash scripts/build.sh
# output: paper/main.pdf
```

Requires a TeX Live installation with `pdflatex` and `bibtex`.

**Regenerate the figures** (from the BNNR repo root, with `bnnr` installed):

```bash
cd ../bnnr
PYTHONPATH=src python3 ../bnnr-research/scripts/generate_figures.py
```

Figures are produced from the same four CC0 demo images used in the
pytorch-grad-cam integration notebook (Labrador, tabby cat, espresso, sports
car), rendered with matplotlib defaults and no branding.

## Structure

```
paper/
  main.tex            root document
  refs.bib            bibliography
  sections/           one .tex per section
  figures/            generated PNG/PDF figures
scripts/
  build.sh            pdflatex + bibtex build
  generate_figures.py figure generation from BNNR
```
