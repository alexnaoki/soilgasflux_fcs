# Processing Notebooks

This folder contains notebooks for turning raw chamber measurements into FCS processing outputs.

## Recommended Notebook

- `01_process_raw_data.ipynb` is the main user-facing workflow. It can load the current JSON-folder format or generic CSV files, map user-specific column names into the canonical FCS schema, preview the data, and run standard FCS processing with optional MCMC.

## Legacy Notebooks

Older raw-processing and summary notebooks are kept in `legacy/` for reference. They may contain machine-specific paths or project-specific assumptions.

## Outputs

Processing outputs are written to `output/` by default. That folder is ignored by git.
