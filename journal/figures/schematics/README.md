# Editable schematic masters

These SVG files are reusable exports of the five conceptual panels in Figure
1. They contain vector geometry and live text, with no embedded raster images.
They can be opened in Adobe Illustrator, Affinity Designer, Inkscape or Figma.

Regenerate them from the shared source:

```bash
cd journal
python scripts/figure1_vector_schematics.py
```

The same drawing functions build the publication-facing Figure 1, preventing
the paper and presentation versions from drifting apart. The canonical
manuscript asset remains `../main/figure_01.pdf`; its editable full-figure
master is `../main/figure_01.svg`.
