# Soil Gas Flux Chamber Simulation - Mathematical Framework

This document describes the mathematical equations and physical processes used in the `synthetic/simulate_sensor.py` module for simulating gas flux measurements in a chamber-sensor system.

---

## Table of Contents
1. [Ideal Gas Law Conversions](#ideal-gas-law-conversions)
2. [Chamber Model (HM Model)](#chamber-model-hm-model)
3. [Chamber Discretization](#chamber-discretization)
4. [Gas Transport Processes](#gas-transport-processes)
5. [Sensor Response Model](#sensor-response-model)
6. [Mass Balance Equations](#mass-balance-equations)

---

## 1. Ideal Gas Law Conversions

### 1.1 Conversion from Volume Concentration (ppm) to Mass Concentration (g/cm³)

The ideal gas law is used to convert between volumetric concentration (ppm) and mass concentration:

$$
\rho_{mass} = \frac{P \cdot C_{ppm} \cdot M}{10^6 \cdot R \cdot T}
$$

Where:
- $\rho_{mass}$ = mass concentration [g/cm³]
- $P$ = pressure [Pa] (default: 101325 Pa)
- $C_{ppm}$ = volumetric concentration [ppm]
- $M$ = molar mass [g/mol] (44.01 g/mol for CO₂)
- $R$ = universal gas constant = 8.3145 × 10⁶ [cm³·Pa·K⁻¹·mol⁻¹]
- $T$ = temperature [K] = $T_{celsius} + 273.15$

**Implementation:** `_idealgaslaw_convertion_to_massConcentration()`

### 1.2 Conversion from Mass Concentration (g/cm³) to Volume Concentration (ppm)

The inverse conversion:

$$
C_{ppm} = \frac{\rho_{mass} \cdot R \cdot T}{P \cdot M} \cdot 10^6
$$

**Implementation:** `_idealgaslaw_convertion_to_ppm()`

---

## 2. Chamber Model (HM Model)

The Hutchinson-Mosier (HM) exponential model describes the temporal evolution of gas concentration in the chamber:

### 2.1 Concentration Evolution

$$
C(t) = C_{\infty} + (C_0 - C_{\infty}) \cdot e^{-\alpha(t - t_0)}
$$

Where:
- $C(t)$ = concentration at time $t$ [ppm]
- $C_{\infty}$ = equilibrium/saturation concentration [ppm]
- $C_0$ = initial concentration [ppm]
- $\alpha$ = rate constant [s⁻¹]
- $t_0$ = initial time [s]
- $e$ = Euler's number (≈ 2.71828)

**Implementation:** `models.hm_model()`

### 2.2 Rate of Concentration Change

The derivative of the HM model gives the rate of concentration change:

$$
\frac{dC}{dt} = \alpha \cdot (C_{\infty} - C_0) \cdot e^{-\alpha(t - t_0)}
$$

**Implementation:** `models.hm_model_dcdt()`

### 2.3 Soil Gas Flux to dC/dt Conversion

Converting from soil gas flux to rate of concentration change:

$$
\frac{dC}{dt} = \frac{F_{soil} \cdot R \cdot A \cdot T}{10 \cdot V \cdot P \cdot (1 - w_0/1000)}
$$

Where:
- $F_{soil}$ = soil gas flux [μmol·m⁻²·s⁻¹]
- $R$ = 8.314 [J·K⁻¹·mol⁻¹]
- $A$ = chamber area [cm²]
- $T$ = temperature [K]
- $V$ = chamber volume [cm³]
- $P$ = pressure [kPa]
- $w_0$ = water vapor mole fraction [mmol/mol]

**Implementation:** `models.dcdt_from_soilgasflux()`

---

## 3. Chamber Discretization

The chamber is discretized into vertical nodes to simulate spatial gradients:

### 3.1 Number of Nodes

$$
N_{nodes} = \lfloor \frac{V_{chamber}}{A_{chamber}} \rfloor
$$

Where:
- $N_{nodes}$ = number of vertical nodes (dimensionless)
- $V_{chamber}$ = chamber volume [cm³]
- $A_{chamber}$ = chamber cross-sectional area [cm²]

This effectively creates nodes with height ≈ 1 cm.

### 3.2 Gas Mass per Node

$$
m_{node,i}(t) = \rho_{mass,i}(t) \cdot \frac{V_{chamber}}{N_{nodes}}
$$

Where:
- $m_{node,i}(t)$ = gas mass in node $i$ at time $t$ [g]
- $\rho_{mass,i}(t)$ = mass concentration in node $i$ [g/cm³]

---

## 4. Gas Transport Processes

### 4.1 Source Input (Soil Emission)

At each time step, gas mass is added to the bottom node (node 0):

$$
m_{source}(\Delta t) = \rho_{source} \cdot V_{total} \cdot \Delta t
$$

$$
m_{node,0}(t+\Delta t) = m_{node,0}(t) + m_{source}(\Delta t)
$$

Where:
- $\Delta t$ = time step [s]
- $\rho_{source}$ = source gas density [g/s]
- $V_{total}$ = total system volume [cm³]

### 4.2 Diffusion Between Nodes

Stochastic diffusion between adjacent nodes:

$$
\Delta m_{diff} = \frac{|m_{node,i}(t) - m_{node,i+1}(t)|}{2} \cdot \xi_{diff}
$$

$$
\text{If } m_{node,i} > m_{node,i+1}:
$$

$$
m_{node,i}(t+\Delta t) = m_{node,i}(t) - \Delta m_{diff}
$$

$$
m_{node,i+1}(t+\Delta t) = m_{node,i+1}(t) + \Delta m_{diff}
$$

Where:
- $\xi_{diff}$ = stochastic diffusion coefficient, uniformly distributed in [0.5, 1.0]
- Diffusion occurs from higher to lower concentration

**Implementation:** Enabled with `with_diffusion=True` in `run_simulation()`

### 4.3 Advective Mixing (Fan)

Chamber mixing by internal fan:

#### Fan Parameters
$$
Q_{fan} = 4000 \text{ cm³/s (default)}
$$

$$
D_{fan} = 4 \text{ cm (fan diameter)}
$$

$$
A_{fan} = \pi \cdot (D_{fan}/2)^2
$$

$$
v_{fan} = \frac{Q_{fan}}{A_{fan}}
$$

$$
\eta_{fan} = 0.5 \text{ (fan efficiency)}
$$

$$
I_{fan} = \frac{Q_{fan}}{V_{chamber}} \cdot \eta_{fan}
$$

Where:
- $Q_{fan}$ = fan flow rate [cm³/s]
- $A_{fan}$ = fan area [cm²]
- $v_{fan}$ = fan velocity [cm/s]
- $\eta_{fan}$ = fan efficiency (dimensionless)
- $I_{fan}$ = fan influence factor (dimensionless)

#### Advective Mass Transport

Gas is redistributed from upper nodes to lower nodes based on fan influence:

$$
V_{influenced} = V_{chamber} \cdot I_{fan}
$$

$$
N_{influenced} = \text{number of nodes where } \sum_{i=0}^{N_{influenced}} V_{node,i} < V_{influenced}
$$

Mass in influenced nodes is redistributed with stochastic weighting:

$$
m_{node,i}^{advected} = \frac{m_{total,influenced}}{\sum \xi_{adv}} \cdot \xi_{adv,i}
$$

Where:
- $\xi_{adv,i}$ = stochastic advection coefficient for node $i$, uniformly distributed in [0.995, 1.0]

**Implementation:** Enabled with `with_advection=True` in `run_simulation()`

### 4.4 Pump Flow (Gas Circulation)

#### Volume and Mass Exchange

At each time step, gas is pumped from the system components:

$$
V_{pumped} = Q_{pump} \cdot \Delta t
$$

Where:
- $Q_{pump}$ = pump rate [cm³/s]

**Pumping sequence:**
1. Empty all non-chamber components (gas analyzer, internal pump, additional volume)
2. If $V_{pumped} > \sum V_{components}$, extract from chamber nodes (starting from top)
3. Calculate density of pumped gas:

$$
\rho_{pumped} = \frac{m_{pumped}}{V_{pumped}}
$$

**Return flow:**
Gas from chamber is redistributed to system components:

$$
m_{component}(t+\Delta t) = \rho_{moved} \cdot V_{component}
$$

---

## 5. Sensor Response Model

### 5.1 Response Time Signal

The sensor has a finite response time, modeled as either linear or logarithmic:

#### Linear Response
$$
C_{response}(t) = C_{start} + (C_{end} - C_{start}) \cdot \frac{t}{t_{response}}
$$

#### Logarithmic Response
$$
C_{response}(t) = C_{start} + (C_{end} - C_{start}) \cdot \frac{\log_{10}(t + 1)}{\log_{10}(t_{response} + 1)}
$$

Where:
- $t_{response}$ = sensor response time [s]

**Implementation:** `response_time_signal()`

### 5.2 Measured Concentration with Sensor Response

The measured concentration accounts for sensor lag and precision:

$$
C_{measured}(t) = \sum_{\tau=0}^{t} S(t, \tau) + C_0 + \mathcal{N}(0, \sigma_{precision})
$$

Where:
- $S(t, \tau)$ = signal matrix accounting for response time
- $\mathcal{N}(0, \sigma_{precision})$ = Gaussian noise with standard deviation $\sigma_{precision}$
- $\sigma_{precision}$ = sensor precision [ppm]

**Implementation:** `sensor_measurement()`

---

## 6. Mass Balance Equations

### 6.1 Total System Mass

The total gas mass in the system is conserved (within numerical precision):

$$
m_{total}(t) = \sum_{i=0}^{N_{nodes}} m_{chamber,i}(t) + m_{analyzer}(t) + m_{pump}(t) + m_{additional}(t)
$$

### 6.2 Mass Balance per Time Step

$$
m_{total}(t+\Delta t) = m_{total}(t) + m_{source}(\Delta t)
$$

This equation verifies conservation of mass in the simulation.

### 6.3 Concentration from Mass

For any component:

$$
C_{component}(t) = \frac{m_{component}(t) / V_{component} \cdot R \cdot T}{P \cdot M} \cdot 10^6
$$

---

## System Parameters

### Default Values
- **Temperature:** 20°C (293.15 K)
- **Pressure:** 101325 Pa (1 atm)
- **CO₂ Molar Mass:** 44.01 g/mol
- **Gas Constant:** 8.3145 × 10⁶ cm³·Pa·K⁻¹·mol⁻¹
- **Fan Flow Rate:** 4000 cm³/s
- **Fan Efficiency:** 0.5

### User-Defined Parameters
- Chamber area and volume
- Gas analyzer volume and response time
- Pump volume and rate
- Sensor accuracy and precision
- Initial concentration $C_0$
- HM model parameters ($\alpha$, $C_{\infty}$, $t_0$)
- Simulation time step $\Delta t$ and total time

---

## Numerical Implementation Notes

1. **Time Discretization:** Forward Euler method with fixed time step $\Delta t$
2. **Spatial Discretization:** Finite volume method with nodes based on chamber height
3. **Stochastic Elements:** 
   - Diffusion: uniform distribution [0.5, 1.0]
   - Advection: uniform distribution [0.995, 1.0]
   - Sensor noise: Gaussian distribution with specified precision
4. **Node Ordering:** Node 0 is at the bottom (near soil), highest node number is at the top (near outlet)

---

## References

- Hutchinson, G. L., & Mosier, A. R. (1981). Improved soil cover method for field measurement of nitrous oxide fluxes. *Soil Science Society of America Journal*, 45(2), 311-316.

---

*Generated for the soilgasflux_fcs simulation package*
