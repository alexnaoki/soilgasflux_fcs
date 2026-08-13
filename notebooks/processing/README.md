# Processing Notebooks

This folder contains notebooks for turning raw chamber measurements into FCS processing outputs.

## Recommended Notebook

- `01_process_raw_data.ipynb` is the main widget-based workflow. It can load the current JSON-folder format or generic CSV files, map user-specific column names into the canonical FCS schema, preview the data, and run standard FCS processing with optional MCMC.
- `02_process_raw_data_plain_python.ipynb` is the coder-friendly workflow. It has no widgets; edit the configuration variables, then run the cells top to bottom.
- `03_process_synthetic_batch.ipynb` processes a matched run from `synthetic_create/03_paper_synthetic_batch.ipynb`. It creates regular, MCMC, and best-Pareto NetCDF files for base, commercial, and low-cost data, preserving the run and scenario metadata in every dataset.
- `04_process_raw_data_with_preprocessing.ipynb` extends the widget workflow with a protected preprocessing copy between column mapping and FCS processing. It supports general acceptable-value ranges and ID-by-ID elapsed-time removal ranges for CO2, pressure, temperature, or humidity; rejected values are interpolated within each measurement ID. Changes can be previewed, accumulated, undone, reset, and optionally exported with an edit log without modifying source files. FCS processing may continue with null pressure, temperature, or humidity values and displays a warning so those environmental gaps can be filled later; null IDs, times, or CO2 values remain blocking.

## Legacy Notebooks

Older raw-processing and summary notebooks are kept in `legacy/` for reference. They may contain machine-specific paths or project-specific assumptions.

## Outputs

Processing outputs are written to `output/` by default. That folder is ignored by git.

The preprocessing workflow exports cleaned canonical CSV files and their edit logs to `output/preprocessed/` by default. The export folder and filename prefix can be changed in the notebook.

Synthetic pipeline outputs use `output/03_synthetic_runs/<run_id>/`. Leave `RUN_ID=None` in the notebook to process the latest generated run, or set an explicit ID to reproduce an earlier run.
