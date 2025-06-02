# soilgasflux_fcs


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
from soilgasflux_fcs import synthetic_create

# Create generator
generator = synthetic_create.Generator(total_time=180, c0=430)

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
from soilgasflux_fcs import simulate_sensor

# Initialize sensor simulator
sim = simulate_sensor.Simulate_Sensor(
    alpha=1e-4, cs=1e4, c0=430, t0=0,
    total_time=120, dt=1
)

# Configure system
sim.chamber_settings(area=314, chamber_volume=6283)
sim.gasAnalyzer_settings(response_time=10, sensor_accuracy=0.1)

# Run simulation
source_flux = np.ones(120) * 1.0  # 1 ppm/s
final_concentration = sim.run_simulation(source_ppm=source_flux)
```

**Configurable Components:**
- Chamber geometry and mixing
- Gas analyzer response time and accuracy
- Internal pumping system
- Environmental conditions

