# Post-Processing Notebooks

This folder contains notebooks for inspecting processed NetCDF outputs, reproducing paper-style Pareto figures, and preserving older exploratory analyses.

## Recommended Notebooks

- `01_browse_processed_netcdf.ipynb` is the main user-facing browser for processed `.nc` files. Use it to select a folder, choose one or more files, inspect the detected schema, and plot individual measurements.
- `02_paper_pareto_lowcost_vs_commercial.ipynb` is the clean paper workflow for synthetic low-cost vs commercial sensor Pareto figures. It expects processed synthetic outputs in `data/` by default, or a user-provided processed-output folder.
- `03_pareto_lowcost_vs_commercial.ipynb` is the run-aware paper workflow for outputs from `processing/03_process_synthetic_batch.ipynb`. It discovers files through the processing manifest, groups scenarios from NetCDF coordinates, and saves the Pareto and split-violin figures under the same run ID.

## Legacy Notebooks

Older exploratory notebooks are kept in `legacy/` for reference.

| Notebook | Notes |
| --- | --- |
| `2-fcs_matrix.ipynb` | Synthetic processed-data matrix exploration. |
| `2.1-fcs_matrix_median.ipynb` | Synthetic processed-data median matrix exploration. |
| `2.2-fcs_pareto.ipynb` | Synthetic low-cost vs commercial Pareto comparison reference. |
| `2.2-fcs_pareto copy.ipynb` | Older copy of the synthetic Pareto comparison. |
| `3-fcs_paretoDistribution.ipynb` | Synthetic Pareto distribution and ridge-plot reference. |
| `4-comparison_between_settings.ipynb` | Synthetic low-cost, commercial, and baseline settings comparison. |
| `plot_processed.ipynb` | General processed-output plotting reference. |
| `plot_processed_filtered.ipynb` | General processed-output plotting with zero-flux filtering. |
| `plot_processed_single_config.ipynb` | General processed-output plotting with a single configuration cell. |

## Data And Figures

- Put local processed `.nc` files in `data/` or point the notebooks to another folder.
- Generated figures are written to `figures/`.
- Stage-3 pipeline figures are written to `figures/03_synthetic_runs/<run_id>/`. Leave `RUN_ID=None` to use the latest processed run.
- Both `data/` and generated PNG figures are ignored by git.
