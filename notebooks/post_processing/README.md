# Post-Processing Notebooks

This folder contains notebooks for inspecting processed NetCDF outputs, reproducing paper-style Pareto figures, and preserving older exploratory analyses.

## Recommended Notebooks

- `01_browse_processed_netcdf.ipynb` is the main user-facing browser for processed `.nc` files. Use it to select a folder, choose one or more files, inspect the detected schema, and plot individual measurements.
- `02_paper_pareto_lowcost_vs_commercial.ipynb` is the clean paper workflow for synthetic low-cost vs commercial sensor Pareto figures. It expects processed synthetic outputs in `data/` by default, or a user-provided processed-output folder.
- `03_pareto_lowcost_vs_commercial.ipynb` is the run-aware paper workflow for outputs from `processing/03_process_synthetic_batch.ipynb`. It discovers files through the processing manifest, groups scenarios from NetCDF coordinates, and saves the Pareto and split-violin figures under the same run ID.
- `04_plot_dcdt_timeseries.ipynb` is the interactive dC/dt timeseries viewer. Enter a processed-results folder, explicitly select standard or best-Pareto MCMC files in separate tabs, filter by timestamp and y range, and optionally apply a time-based moving-window mean. Repeated timestamps are merged with a visible warning. Standard files expose deadband and cutoff selectors; MCMC files show the posterior median with a 16–84% interval.
- `05_calculate_soil_gas_flux.ipynb` is the interactive flux-conversion workflow. Select one dC/dt NetCDF file or a folder of Standard or best-Pareto MCMC results, scan the auxiliary files produced by `processing/05_prepare_auxiliary_flux_data.ipynb`, review exact chamber/timestamp matches, and calculate flux with first-sample or elapsed-window-mean auxiliary values. An optional, explicitly selected donor chamber can supply the closest valid environmental tuple within a configurable time gap while the target chamber geometry is retained. Select one or more calculated files to overlay in the flux preview; Standard overlays use shared deadband/cutoff coordinates, while MCMC overlays show each median and uncertainty band. Previews also include timestamp and y-range controls, optional filtering outside the y range, and an optional time-based moving-window mean; these preview controls never change exports. New `_with_flux.nc` files preserve the original dC/dt data and add flux, conversion inputs, quality flags, and full auxiliary provenance.

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
- Flux-conversion NetCDF files are written to `output/flux/` by default. Source dC/dt and auxiliary files are never modified.
