# soilgasflux_fcs


# Package
## 1. Flux Calculation Schema 

The Flux Calculation Schema (FCS) is the core analytical framework for processing chamber-based soil gas flux measurements. The system implements multiple mathematical models to extract accurate flux estimates from CO₂ concentration time series data, with robust uncertainty quantification through Monte Carlo Markov Chain (MCMC) sampling.

### 1.1. Processing Data

The FCS system processes raw sensor data through a comprehensive workflow that includes data loading, preprocessing, temporal windowing, and model fitting across multiple parameter combinations.

#### 1.1.1. Data Input and Structure

The system accepts time series data containing the following required fields:

```python
from soilgasflux_fcs import FCS, json_reader

# Load data from JSON files
initializer = json_reader.Initializer(folderPath='path/to/data/')
df = initializer.prepare_rawdata()

# Initialize FCS with measurement data
fcs = FCS(df_data=df, chamber_id='chamber_01')
```

**Required Data Fields:**
- `timedelta`: Time elapsed since measurement start (seconds)
- `k30_co2`: CO₂ concentration measurements (ppm)
- `si_temperature`: Chamber temperature (°C)
- `si_humidity`: Relative humidity (%)
- `bmp_pressure`: Atmospheric pressure (Pa)

#### 1.1.2. Moving Window Analysis

The FCS implements a moving window approach to systematically evaluate different temporal segments of the measurement data:

```python
# Configure analysis parameters
fcs.settings(
    moving_window=True,
    window_walk=10,           # Window step size (seconds)
    min_window_size=30,       # Minimum analysis window (seconds)
    min_deadband=0,           # Minimum deadband start (seconds)
    max_deadband=60           # Maximum deadband end (seconds)
)
```

**Window Parameters:**
- **Deadband**: Initial stabilization period after chamber closure
- **Cutoff**: End time for analysis window
- **Window Walk**: Step size for systematic parameter exploration
- **Minimum Window Size**: Ensures sufficient data points for reliable fitting

#### 1.1.3. Data Preprocessing

The system performs several preprocessing steps:

1. **Water Vapor Correction**: Calculates mole fraction of water vapor using Buck's equation
2. **Initial Concentration Estimation**: Determines C₀ from first 10 data points using linear regression
3. **Quality Control**: Validates data completeness and temporal consistency

#### 1.1.4. Metadata Integration

Chamber physical properties are integrated for accurate flux calculations:

```python
metadata = {
    'area': 314,     # Chamber surface area (cm²)
    'volume': 6283   # Chamber volume (cm³)
}

results = fcs.run(n='measurement_id', metadata=metadata)
```

### 1.2. Models

The FCS implements two complementary mathematical models for soil gas flux estimation, each with specific advantages and applications.

#### 1.2.1. Hutchinson-Mosier (H-M) Model

The H-M model accounts for non-linear concentration changes due to finite chamber effects and provides the most physically realistic representation of chamber dynamics.

##### Mathematical Foundation

The H-M model is based on the exponential approach to steady-state:

```
C(t) = Cx + (C₀ - Cx) × exp(-α(t - t₀))
```

Where:
- `C(t)`: CO₂ concentration at time t (ppm)
- `Cx`: Steady-state concentration (ppm)
- `C₀`: Initial concentration (ppm)
- `α`: Chamber-dependent parameter (s⁻¹)
- `t₀`: Time offset parameter (s)

##### Flux Calculation

The instantaneous flux rate is derived from the concentration gradient:

```
dC/dt = α(Cx - C₀) × exp(-α(t - t₀))
```

The soil gas flux is calculated using the ideal gas law:

```
F = (10 × V × P₀ × (1 - W₀/1000) × dC/dt) / (R × A × (T₀ + 273.15))
```

Where:
- `F`: Soil gas flux (μmol m⁻² s⁻¹)
- `V`: Chamber volume (cm³)
- `P₀`: Atmospheric pressure (kPa)
- `W₀`: Water vapor mole fraction (mmol mol⁻¹)
- `R`: Ideal gas constant (8.314 J K⁻¹ mol⁻¹)
- `A`: Chamber area (cm²)
- `T₀`: Temperature (°C)

##### Implementation and Fitting

```python
from soilgasflux_fcs import HM_model

# Initialize H-M model
hm = HM_model(raw_data=df, metadata=metadata)

# Standard analysis
dc_dt, C_0, cx, a, t0, flux, deadband, cutoff = hm.calculate(
    deadband=10, cutoff=120
)

# Monte Carlo uncertainty analysis
results_mc = hm.calculate_MC(deadband=10, cutoff=120, n=2000)
```

**Parameter Estimation:**
- Uses `lmfit` for robust non-linear optimization
- Implements physically meaningful parameter bounds
- Handles fitting convergence issues with error handling

**Advantages:**
- Physically realistic representation of chamber dynamics
- Accounts for non-linear concentration changes
- Provides insights into chamber mixing characteristics

**Limitations:**
- More sensitive to measurement noise
- Requires longer measurement periods for reliable fitting
- Can exhibit convergence issues with poor quality data

#### 1.2.2. Linear Model

The linear model assumes a constant flux rate throughout the measurement period, providing a simpler and more robust alternative for certain applications.

##### Mathematical Foundation

The linear model assumes concentration changes linearly with time:

```
C(t) = dC/dt × t + C₀
```

Where:
- `dC/dt`: Constant concentration change rate (ppm s⁻¹)
- `C₀`: Initial concentration (ppm, fixed parameter)

##### Implementation

```python
from soilgasflux_fcs import LINEAR_model

# Initialize linear model
linear = LINEAR_model(raw_data=df, metadata=metadata)

# Standard analysis
dc_dt, C_0, flux, deadband, cutoff = linear.calculate(
    deadband=10, cutoff=120
)

# Monte Carlo uncertainty analysis
results_mc = linear.calculate_MC(deadband=10, cutoff=120, n=1000)
```

**Parameter Estimation:**
- Single parameter fitting (dC/dt)
- Fixed initial concentration (C₀)
- Gaussian uncertainty propagation

**Advantages:**
- Simple and robust fitting
- Less sensitive to measurement noise
- Reliable for short measurement periods
- Faster computation

**Limitations:**
- Ignores chamber mixing effects
- May overestimate flux for long measurement periods
- Less physically realistic for non-linear processes

#### 1.2.3. Model Comparison and Selection

The FCS automatically calculates multiple performance metrics for model comparison:

```python
# Both models run simultaneously
results = fcs.run(n='measurement_id', metadata=metadata)

# Access model-specific results
hm_flux = results['measurement_id']['dcdt(HM)']
linear_flux = results['measurement_id']['dcdt(linear)']

# Performance metrics
hm_aic = results['measurement_id']['AIC(HM)']
hm_rmse = results['measurement_id']['RMSE(HM)']
hm_r2 = results['measurement_id']['R2(HM)']
```

**Performance Metrics:**
- **AIC (Akaike Information Criterion)**: Model selection with penalty for complexity
- **RMSE (Root Mean Square Error)**: Absolute fit quality
- **R²**: Coefficient of determination
- **nRMSE (Normalized RMSE)**: Scale-independent error metric

### 1.3. Uncertainty Quantification

#### 1.3.1. Monte Carlo Markov Chain (MCMC) Analysis

The FCS implements Bayesian parameter estimation using the `emcee` ensemble sampler for robust uncertainty quantification:

```python
# Run MCMC analysis
results_mc = fcs.run_MC(n='measurement_id', n_MC=2000, metadata=metadata)
```

**MCMC Configuration:**
- **Walkers**: 100 ensemble walkers
- **Steps**: 2000 iterations per chain
- **Burn-in**: 80% of steps discarded
- **Thinning**: Every 15th sample retained

**Prior Distributions:**
- Logarithmic uniform priors for α and Cx parameters
- Uniform prior for t₀ parameter
- Physically motivated parameter bounds

**Likelihood Function:**
```python
ln_likelihood = -0.5 × Σ((y - model)² / σ² + ln(2πσ²))
```

#### 1.3.2. Parameter Uncertainty Propagation

Monte Carlo sampling provides comprehensive uncertainty estimates:

```python
# Extract uncertainty bounds
dcdt_median = np.median(results_mc['dcdt(HM)'])
dcdt_lower = np.percentile(results_mc['dcdt(HM)'], 16)
dcdt_upper = np.percentile(results_mc['dcdt(HM)'], 84)
```

### 1.4. Multiprocessing Framework

The FCS supports parallel processing for large datasets using the `multiprocess_raw_data` module:

```python
from soilgasflux_fcs import multiprocess_raw_data

# Initialize multiprocessor
processor = multiprocess_raw_data.Multiprocessor()

# Parallel processing
results = processor.run(
    df=df, 
    chamber_id='site_01',
    output_folder='./output'
)

# Monte Carlo parallel processing
results_mc = processor.run_MC(
    df=df,
    chamber_id='site_01_MC', 
    output_folder='./output'
)
```

**Features:**
- Automatic CPU core detection and utilization
- Daily batch processing for temporal organization
- NetCDF output format with xarray integration
- Memory-efficient processing of large datasets

**Output Structure:**
```python
# 3D arrays: [time, cutoff, deadband] or [time, cutoff, deadband, MC]
output_vars = [
    'dcdt(HM)', 'dcdt(linear)',           # Flux estimates
    'AIC(HM)', 'AIC(linear)',             # Model selection criteria
    'RMSE(HM)', 'RMSE(linear)',           # Fit quality
    'R2(HM)', 'R2(linear)',               # Correlation metrics
    'nRMSE(HM)', 'nRMSE(linear)',         # Normalized errors
    'logprob(HM)'                         # MCMC log probabilities
]
```

### 1.5. Performance Metrics and Evaluation

#### 1.5.1. Statistical Metrics

The `metrics` module provides comprehensive model evaluation tools:

```python
from soilgasflux_fcs import metrics

# Calculate performance metrics
aic = metrics.calculate_AIC(y_observed, y_predicted, n_parameters=5)
rmse = metrics.rmse(y_observed, y_predicted)
r2 = metrics.r2(y_observed, y_predicted)
nrmse = metrics.normalized_rmse(y_observed, y_predicted)
```

#### 1.5.2. Minimum Detectable Flux

The system includes tools for assessing measurement detection limits:

```python
# Calculate minimum detectable flux
mdf = metrics.minimum_detectable_flux(
    Aa=2.0,          # Analytical accuracy (ppb)
    tc=120,          # Closure time (s)
    freq=1.0,        # Measurement frequency (Hz)
    V=0.006283,      # Volume (m³)
    A=0.0314,        # Area (m²)
    P=101325,        # Pressure (Pa)
    T=298.15         # Temperature (K)
)
```

### 1.6. Workflow Integration

#### 1.6.1. Complete Analysis Pipeline

```python
# Complete FCS workflow
from soilgasflux_fcs import json_reader, multiprocess_raw_data

# 1. Data loading
initializer = json_reader.Initializer('data/raw/')
df = initializer.prepare_rawdata()

# 2. Parallel processing
processor = multiprocess_raw_data.Multiprocessor()
results = processor.run_MC(df, 'site_analysis', './output')

# 3. Results analysis
import xarray as xr
ds = xr.open_dataset('./output/site_analysis_2024-01-01.nc')

# Extract optimal parameters based on AIC
best_params = ds.isel(AIC=ds['AIC(HM)'].argmin(dim=['cutoff', 'deadband']))
```

#### 1.6.2. Quality Assessment

The FCS provides multiple approaches for quality assessment:

1. **Model Convergence**: Parameter uncertainty estimates
2. **Fit Quality**: RMSE and R² metrics
3. **Model Selection**: AIC-based comparison
4. **Physical Plausibility**: Parameter range validation

This comprehensive framework ensures robust and reliable soil gas flux estimation with full uncertainty quantification, suitable for both research applications and operational monitoring systems.

## 2. Synthetic Data Generation and Sensor Simulation

The `soilgasflux_fcs` package provides comprehensive tools for generating synthetic soil gas flux data and simulating realistic sensor behavior. This functionality is essential for validating flux calculation methods, testing algorithm performance, and understanding measurement uncertainties under controlled conditions.

### 2.1. Synthetic Data Generator (`synthetic_create.py`)

The `Generator` class creates realistic synthetic chamber measurements based on the Hutchinson-Mosier model with configurable parameters and disturbances.

#### 2.1.1. Basic Usage

```python
from soilgasflux_fcs import synthetic_create

# Initialize generator
generator = synthetic_create.Generator(total_time=180, c0=430)

# Explore parameter space
generator.alpha_cs_plot(alpha_min=1e-5, alpha_max=1e-2,
                       cs_min=1e3, cs_max=1e6, n=100)
```

#### 2.1.2. Parameter Space Exploration

The generator allows systematic exploration of the α (chamber-dependent parameter) and Cs (steady-state concentration) parameter space:

- **α range**: Controls the non-linearity of concentration changes (typically 1e-5 to 1e-2)
- **Cs range**: Steady-state concentration reached at infinite time (typically 1e3 to 1e6 ppm)
- **Visualization**: Contour plots showing concentration evolution, flux rates, and curvature characteristics

#### 2.1.3. Curve Type Selection

The generator categorizes synthetic curves based on their curvature characteristics:

```python
# Generate curves with different dcdt characteristics
for dcdt in [0.1, 0.5, 1.0]:
    generator.cc_curve_plot(selected_dcdt=dcdt)
```

**Curve Types:**
- **Big Curve**: High curvature, significant chamber effects
- **Straight Curve**: Low curvature, minimal chamber effects  
- **Intermediate**: Between the two extremes

#### 2.1.4. Noise and Disturbance Simulation (Deprecated)

The generator can add realistic measurement disturbances:

```python
# Generate synthetic data with noise
config = generator.generate_base(
    alpha=1e-4, cs=1e4, c0=430, t0=0,
    total_time=180, deadband=10,
    add_noise=True, noise_intensity=2.0,
    noise_type='exp'  # Exponential noise profile
)
```

**Disturbance Parameters:**
- **Deadband**: Initial stabilization period with background concentration
- **Background band**: Periods of environmental interference
- **Unmixed phase**: Chamber closure disturbances
- **Mixed phase disturbance**: Fan-induced mixing effects
- **Noise types**: 'exp' (exponential decay) or None (Gaussian)

#### 2.1.5. Batch Generation

```python
# Create multiple synthetic datasets
generator.create_selected(
    add_noise=True, 
    noise_intensity=1.5,
    noise_type='exp',
    save_path='/path/to/synthetic/data/'
)
```

### 2.2. Sensor Simulation (`simulate_sensor.py`)

The `Simulate_Sensor` class provides detailed simulation of chamber measurement systems including physical gas transport, sensor response, and measurement artifacts.

#### 2.2.1. Basic Sensor Setup

```python
from soilgasflux_fcs import simulate_sensor

# Initialize sensor simulator
sim = simulate_sensor.Simulate_Sensor(
    alpha=1e-4, cs=1e4, c0=430, t0=0, 
    total_time=120, dt=1,
    temperature=20, pressure=101325
)

# Configure chamber geometry
sim.chamber_settings(
    area=np.pi*20**2/4,      # Chamber area (cm²)
    chamber_volume=np.pi*20**2/4*20  # Chamber volume (cm³)
)
```

#### 2.2.2. System Components

**Internal Pump System:**
```python
sim.internal_pump_settings(
    pump_volume=1,      # Internal volume (cm³)
    pump_rate=250       # Flow rate (cm³/s)
)
```

**Gas Analyzer Configuration:**
```python
sim.gasAnalyzer_settings(
    response_time=10,           # Sensor response time (s)
    gasAnalyzer_volume=30,      # Analyzer volume (cm³)
    sensor_accuracy=0.1,        # Measurement accuracy (ppm)
    sensor_precision=0.5        # Measurement precision (ppm)
)
```

**Additional System Volume:**
```python
sim.additional_settings(volume=20)  # Tubing, connectors, etc. (cm³)
```

#### 2.2.3. Physical Process Simulation

The simulator models realistic physical processes:

**Gas Transport Mechanisms:**
- **Diffusion**: Concentration gradient-driven mixing between chamber nodes
- **Advection**: Fan-induced bulk gas movement and mixing
- **Pumping**: Gas circulation through analyzer and back to chamber

**Chamber Discretization:**
- Chamber divided into nodes based on geometry (volume/area ratio)
- Each node tracks gas mass and concentration independently
- Mass conservation maintained throughout system

#### 2.2.4. Running Simulations

```python
# Define source flux profile
source_ppm = np.ones(total_time) * 1.0  # Constant 1 ppm/s input

# Run comprehensive simulation
final_concentration = sim.run_simulation(
    source_ppm=source_ppm,
    with_diffusion=True,    # Enable diffusive mixing
    with_advection=True,    # Enable fan mixing
    verbose=True           # Detailed output
)
```

#### 2.2.5. Sensor Response Modeling

The simulator includes realistic sensor characteristics:

**Response Time Effects:**
- Linear or logarithmic response time profiles
- Configurable sampling frequency
- Signal integration over response time window

**Measurement Uncertainties:**
- Accuracy: Systematic offset from true concentration
- Precision: Random measurement noise
- Environmental drift effects

### 2.3. Sensor Type Comparison

The package enables comparison of different sensor technologies:

#### 2.3.1. Commercial-Grade Sensors
```python
# High-end commercial sensor
sim.gasAnalyzer_settings(
    response_time=1,            # Fast response
    gasAnalyzer_volume=30,
    sensor_accuracy=0.1,        # High accuracy
    sensor_precision=0.5        # Good precision
)
```

#### 2.3.2. Low-Cost Sensors
```python
# Low-cost sensor simulation
sim.gasAnalyzer_settings(
    response_time=20,           # Slower response
    gasAnalyzer_volume=30,
    sensor_accuracy=0.1,        # Maintained accuracy
    sensor_precision=1.5        # Lower precision
)
```

### 2.4. Output Format

Synthetic data is saved in JSON format compatible with the main processing pipeline:

```json
{
  "raw_data": {
    "datetime": [0, 1, 2, ...],
    "datetime_utc": ["2024-01-01 10:00:00", ...],
    "k30_co2": [430.1, 432.3, 435.8, ...],
    "bmp_pressure": [99000, 99000, ...],
    "bmp_temperature": [20.0, 20.0, ...],
    "si_humidity": [70.0, 70.0, ...],
    "si_temperature": [20.0, 20.0, ...]
  },
  "config": {
    "alpha": 1e-4,
    "c_s": 1e4,
    "c_c0": 430,
    "total_time": 180,
    "deadband": 10,
    "add_noise": true,
    "noise_intensity": 1.5,
    "curvature": "big"
  }
}
```

This synthetic data can then be processed using the same workflow as real measurements, enabling direct validation of flux calculation methods.

