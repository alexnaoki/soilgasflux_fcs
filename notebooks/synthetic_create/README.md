# Synthetic Notebooks

This folder keeps synthetic data creation and visualization separate from the FCS processing notebooks.

## Notebooks

- `01_paper_synthetic_figure.ipynb` regenerates the synthetic curves used for the paper-style figure and saves the output to `figures/synthetic_data.png`. The perfect curve is generated from the HM model; the commercial and low-cost traces are produced with `synthetic.simulate_sensor.Simulate_Sensor`.
- `02_interactive_synthetic_creator.ipynb` provides both plain Python functions and an `ipywidgets` interface for creating synthetic JSON files with selectable sensor presets.
- `03_paper_synthetic_batch.ipynb` creates the same paper-style figure together with a matched six-scenario JSON batch for the base, commercial, and low-cost variants. Each execution is stored under a unique run ID for the processing and Pareto notebooks.

## Outputs

- Generated JSON files are written to `generated/` by default.
- Matched Stage-3 pipeline batches are written to `generated/03_runs/<run_id>/`, with a `manifest.json` that records scenario metadata and file paths.
- Generated figures are written to `figures/`.
- `generated/` is ignored by git. The reference `figures/synthetic_data.png` remains tracked as the current paper-figure target.
