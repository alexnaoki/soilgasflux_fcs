# soilgasflux_fcs

A Python package for processing chamber CO₂ measurements to calculate soil gas flux using advanced mathematical models with uncertainty quantification.

## Requirements

### System Requirements
- Python 3.9 or higher
- Operating System: Windows, macOS, or Linux
- Recommended: 8+ GB RAM for large datasets with MCMC analysis

### Python Dependencies

#### Core Dependencies
```
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.5.0
scipy >= 1.7.0
xarray >= 0.19.0
```

#### Optional Dependencies
```
emcee >= 3.0.0          # For MCMC uncertainty quantification
pathlib                 # For file path handling (included in Python 3.4+)
json                     # For data input/output (standard library)
multiprocessing          # For parallel processing (standard library)
```

#### Development Dependencies
```
jupyter                  # For running example notebooks
netcdf4                  # For NetCDF file support with xarray
```

## Installation

### Recommended terminal installation (pip only)

Clone the repository, enter its folder, and source the provided installation file:

```bash
git clone https://github.com/alexnaoki/soilgasflux_fcs.git
cd soilgasflux_fcs
source installation.txt
```

`installation.txt` is intended for macOS and Linux terminals using Bash or Zsh. It creates `.venv`, activates it in the current terminal, upgrades pip, and installs the package with its notebook/development dependencies. It does not use Conda, Mamba, or another package manager.

When returning to the project later, reactivate the environment with:

```bash
source .venv/bin/activate
```

For Windows PowerShell, perform the equivalent pip-only installation manually:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

All package dependencies are declared in `pyproject.toml` and installed automatically. Python 3.9 or newer, including its standard `venv` module, must already be available.

### Conda installation

If Conda is already installed, clone the repository and source the Conda installer from its root:

```bash
git clone https://github.com/alexnaoki/soilgasflux_fcs.git
cd soilgasflux_fcs
source installation_conda.txt
```

This creates or updates the `soilgasflux_fcs` environment from `environment.yml`, activates it in the current Bash or Zsh terminal, and verifies that the package imports successfully. Conda installs the scientific and notebook dependencies from `conda-forge`; pip is used only inside that managed environment to install this local repository as an editable package.

Reactivate the environment later with:

```bash
conda activate soilgasflux_fcs
```

From Windows Anaconda Prompt or PowerShell, run:

```powershell
conda env create --file environment.yml
conda activate soilgasflux_fcs
python -c "import soilgasflux_fcs; print('soilgasflux_fcs is installed')"
```

If the environment already exists and `environment.yml` has changed, synchronize it with `conda env update --file environment.yml --prune`.

### Package layout

FCS processing code lives in the `soilgasflux_fcs` package. Synthetic data helpers live separately in the top-level `synthetic` package so generated test-data workflows are kept apart from production processing code:

```python
# Flat (notebooks use this)
from soilgasflux_fcs import FCS, HM_model, LINEAR_model, Pareto, json_reader

# Organized
from soilgasflux_fcs.core import FCS, HM_model, LINEAR_model
from soilgasflux_fcs.analysis import Pareto
from soilgasflux_fcs.io import Initializer, Multiprocessor
from synthetic import Generator, Simulate_Sensor
```

### Logging

The package uses the standard `logging` module. Default level is `WARNING`. Enable per-fit diagnostics with:

```python
import logging
logging.getLogger('soilgasflux_fcs').setLevel(logging.DEBUG)
```

# Package
## 1. Flux Calculation Schema 

The FCS processes chamber CO₂ measurements to calculate soil gas flux using two mathematical models with uncertainty quantification.

### 1.1. Quick Start

```python
from soilgasflux_fcs import FCS, json_reader

# Load data
initializer = json_reader.Initializer(folderPath='path/to/data/')
df = initializer.prepare_rawdata()

# Initialize and configure
fcs = FCS(df_data=df, chamber_id='chamber_01')
fcs.settings(moving_window=True, window_walk=10, min_window_size=30)

# Run analysis
metadata = {'area': 314, 'volume': 6283}  # cm²/cm³
results = fcs.run(n='measurement_id', metadata=metadata)
```

**Required data fields:** `timedelta`, `k30_co2`, `si_temperature`, `si_humidity`, `bmp_pressure`

### 1.2. The Hutchinson-Mosier (H-M) Model

The **Hutchinson-Mosier model** is the primary approach for calculating soil CO2 flux from chamber measurements. It models the exponential accumulation of CO2 in a closed chamber system:

**C(t) = C_x + (C_0 - C_x) × exp(-α × (t - t_0))**

Where:
- `C(t)` = CO2 concentration at time t
- `C_x` = Equilibrium CO2 concentration  
- `C_0` = Initial CO2 concentration
- `α` (alpha) = Rate parameter related to soil flux
- `t_0` = Time offset parameter

```python
from soilgasflux_fcs import HM_model

hm = HM_model(raw_data=df, metadata=metadata)
dc_dt, C_0, cx, a, t0, flux, deadband, cutoff = hm.calculate(deadband=10, cutoff=120)

# With uncertainty quantification via MCMC
results_mc = hm.calculate_MC(deadband=10, cutoff=120, n=2000)
```

**When to use:** The H-M model is recommended for most soil CO2 flux measurements as it:
- Accounts for the exponential nature of CO2 accumulation in closed chambers
- Provides physically meaningful parameters  
- Handles measurement uncertainties through MCMC sampling
- Works well with typical 3-5 minute chamber deployment periods

**Alternative:** A simple linear model is available as a fallback option for cases where the H-M model fails to converge or for very short measurement periods.

### 1.3. Pareto Front Analysis for Optimal Parameter Selection

The package includes advanced model selection using Pareto front analysis to find the optimal balance between uncertainty and model fit quality:

```python
import numpy as np

def find_pareto_front(x, y, maximize_x=False, maximize_y=False):
    """
    Find the Pareto front for two objectives
    
    Parameters:
    -----------
    x, y : array-like
        Values of the two objectives (uncertainty range vs loglikelihood)
    maximize_x, maximize_y : bool
        Whether to maximize (True) or minimize (False) each objective
    
    Returns:
    --------
    pareto_indices : ndarray
        Indices of points on the Pareto front
    """
    points = np.column_stack((x if not maximize_x else -x, y if not maximize_y else -y))
    pareto_indices = []
    
    for i, point in enumerate(points):
        if np.isnan(point).any():
            continue
        dominated = False
        for j, other_point in enumerate(points):
            if i != j and not np.isnan(other_point).any():
                if (all(other_point <= point) and any(other_point < point)):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    return np.array(pareto_indices)

# Example usage with MCMC results
# Calculate uncertainty range (68% confidence interval)
uncertainty_range = np.quantile(results_mc['dcdt_samples'], 0.84) - np.quantile(results_mc['dcdt_samples'], 0.16)

# Get log-likelihood values from MCMC
loglikelihood = -results_mc['logprob_samples'].mean()

# Find optimal parameters using Pareto front
# Minimize both uncertainty (x) and negative log-likelihood (y)
pareto_indices = find_pareto_front(uncertainty_range, loglikelihood, maximize_x=False, maximize_y=False)

# Select best compromise solution (closest to origin)
distances = np.sqrt(uncertainty_range[pareto_indices]**2 + loglikelihood[pareto_indices]**2)
best_idx = pareto_indices[np.argmin(distances)]
```

This analysis helps you:
- **Balance precision vs. confidence:** Lower uncertainty vs. better model fit
- **Optimize measurement windows:** Find ideal deadband/cutoff combinations
- **Quality control:** Identify problematic measurements with poor trade-offs
- **Parameter selection:** Choose optimal MCMC chain parameters

### 1.4. Models

#### 1.4.1. Linear Model (Alternative)

Assumes constant flux rate with simple linear fitting.

```python
from soilgasflux_fcs import LINEAR_model

linear = LINEAR_model(raw_data=df, metadata=metadata)
dc_dt, C_0, flux, deadband, cutoff = linear.calculate(deadband=10, cutoff=120)
```

**Use when:** H-M model fails to converge or for quick analysis of very short measurement periods.

#### 1.4.2. Model Comparison

```python
# Both models run automatically
results = fcs.run(n='measurement_id', metadata=metadata)

# Compare results using AIC (Akaike Information Criterion)
hm_flux = results['measurement_id']['dcdt(HM)']
linear_flux = results['measurement_id']['dcdt(linear)']  
hm_aic = results['measurement_id']['AIC(HM)']  # Lower AIC = better model
linear_aic = results['measurement_id']['AIC(linear)']
```

### 1.5. Parallel Processing

For large datasets, use multiprocessing:

```python
from soilgasflux_fcs import multiprocess_raw_data

processor = multiprocess_raw_data.Multiprocessor()

# Standard analysis
results = processor.run(df=df, chamber_id='site_01', output_folder='./output')

# With uncertainty quantification
results_mc = processor.run_MC(df=df, chamber_id='site_01_MC', output_folder='./output')
```

Output saved as NetCDF files with dimensions: `[time, cutoff, deadband]` or `[time, cutoff, deadband, MC]`

### 1.6. Complete Workflow Example

```python
from soilgasflux_fcs import json_reader, multiprocess_raw_data
import xarray as xr

# 1. Load data
initializer = json_reader.Initializer('data/raw/')
df = initializer.prepare_rawdata()

# 2. Process with uncertainty
processor = multiprocess_raw_data.Multiprocessor()
results = processor.run_MC(df, 'analysis', './output')

# 3. Analyze results
ds = xr.open_dataset('./output/analysis_2024-01-01.nc')
best_params = ds.isel(AIC=ds['AIC(HM)'].argmin(dim=['cutoff', 'deadband']))
flux_median = ds['dcdt(HM)'].median(dim='MC')
flux_std = ds['dcdt(HM)'].std(dim='MC')
```

## 2. Synthetic Data Generation

Generate synthetic chamber measurements for testing and validation.

### 2.1. Basic Synthetic Data

```python
from synthetic import Generator

# Create generator
generator = Generator(total_time=180, c0=430)

# Generate synthetic measurement
config = generator.generate_base(
    alpha=1e-4, cs=1e4, c0=430, t0=0,
    total_time=180, deadband=10,
    add_noise=True, noise_intensity=2.0
)

# Create multiple datasets
generator.create_selected(
    add_noise=True,
    noise_intensity=1.5,
    save_path='/path/to/synthetic/data/'
)
```

### 2.2. Sensor Simulation

Simulate realistic sensor response including physical transport and measurement artifacts.

```python
import numpy as np
from synthetic import simulate_sensor

# Initialize sensor simulator
sim = simulate_sensor.Simulate_Sensor(
    alpha=1e-4, cs=1e4, c0=430, t0=0,
    total_time=120, dt=1
)

# Configure system
sim.chamber_settings(area=314, chamber_volume=6283)
sim.gasAnalyzer_settings(
    response_time=10,
    gasAnalyzer_volume=50,
    sensor_accuracy=0.1,
    sensor_precision=0.5,
)
sim.internal_pump_settings(pump_volume=25, pump_rate=250)
sim.additional_settings(volume=25)

# Run simulation
source_flux = np.ones(120) * 1.0  # 1 ppm/s
final_concentration = sim.run_simulation(source_ppm=source_flux)
```

**Configurable Components:**
- Chamber geometry and mixing
- Gas analyzer response time and accuracy
- Internal pumping system
- Environmental conditions
