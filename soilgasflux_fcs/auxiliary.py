'''
Per-measurement quality control of auxiliary flux data (pressure, temperature,
relative humidity and chamber geometry), used by
notebooks/processing/05_prepare_auxiliary_flux_data.ipynb.

It lives in the package rather than the notebook because multiprocessing
workers (spawned on macOS and Windows) can only run importable functions.
'''
import math
import multiprocessing as mp

import numpy as np
import pandas as pd

from .models import mole_fraction_water_vapor

GROUP_KEYS = ["chamber_id", "measurement_id"]

ENVIRONMENT_SPECS = {
    "pressure_kpa": {
        "label": "Pressure",
        "unit": "kPa",
        "source": "pressure_source",
        "flag": "pressure_quality_flag",
    },
    "temperature_c": {
        "label": "Temperature",
        "unit": "degree_Celsius",
        "source": "temperature_source",
        "flag": "temperature_quality_flag",
    },
    "relative_humidity_percent": {
        "label": "Relative humidity",
        "unit": "percent",
        "source": "relative_humidity_source",
        "flag": "relative_humidity_quality_flag",
    },
}

# Below this many measurements, starting worker processes (about 3 s, mostly
# importing the package) costs more time than it saves, so they are prepared
# in the calling process.
MIN_PARALLEL_MEASUREMENTS = 2000


def interpolate_environment_signal(elapsed, values, lower, upper):
    elapsed = np.asarray(elapsed, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= lower) & (values <= upper)
    filled = np.full(values.shape, np.nan, dtype=float)
    flags = np.full(values.shape, 2, dtype=np.int8)

    if valid.any():
        order = np.argsort(elapsed, kind="stable")
        sorted_elapsed = elapsed[order]
        anchor_elapsed = elapsed[valid]
        anchor_values = values[valid]
        anchor_order = np.argsort(anchor_elapsed, kind="stable")
        interpolated = np.interp(
            sorted_elapsed,
            anchor_elapsed[anchor_order],
            anchor_values[anchor_order],
        )
        filled[order] = interpolated
        flags[:] = 1
        flags[valid] = 0

    return filled, flags


def geometry_quality_flags(values, default_used):
    numeric = np.asarray(values, dtype=float)
    valid = np.isfinite(numeric) & (numeric > 0)
    if default_used:
        flags = np.where(valid, 1, 2)
    else:
        flags = np.where(valid, 0, 2)
    return flags.astype(np.int8)


def prepare_measurement(group, ranges):
    '''
    Fill the environmental signals of one chamber/measurement group, derive
    water vapour, and set the quality flags and measurement status.
    '''
    prepared_group = group.copy().sort_values("timestamp")
    elapsed = prepared_group["elapsed_seconds"].to_numpy(dtype=float)

    for output_name, spec in ENVIRONMENT_SPECS.items():
        filled, flags = interpolate_environment_signal(
            elapsed,
            prepared_group[spec["source"]].to_numpy(dtype=float),
            *ranges[output_name],
        )
        prepared_group[output_name] = filled
        prepared_group[spec["flag"]] = flags

    area_default = bool(prepared_group["area_default_used"].iloc[0])
    volume_default = bool(prepared_group["volume_default_used"].iloc[0])
    prepared_group["chamber_area_quality_flag"] = geometry_quality_flags(
        prepared_group["chamber_area_cm2"], area_default
    )
    prepared_group["chamber_volume_quality_flag"] = geometry_quality_flags(
        prepared_group["chamber_volume_cm3"], volume_default
    )

    valid_water = (
        prepared_group["pressure_quality_flag"].ne(2)
        & prepared_group["temperature_quality_flag"].ne(2)
        & prepared_group["relative_humidity_quality_flag"].ne(2)
    )
    water = np.full(len(prepared_group), np.nan, dtype=float)
    if valid_water.any():
        indexes = valid_water.to_numpy()
        water[indexes] = mole_fraction_water_vapor(
            prepared_group.loc[valid_water, "temperature_c"].to_numpy(dtype=float),
            prepared_group.loc[valid_water, "relative_humidity_percent"].to_numpy(dtype=float),
            prepared_group.loc[valid_water, "pressure_kpa"].to_numpy(dtype=float),
        )
    prepared_group["water_vapor_mmol_mol"] = water

    original_inputs = (
        prepared_group["pressure_quality_flag"].eq(0)
        & prepared_group["temperature_quality_flag"].eq(0)
        & prepared_group["relative_humidity_quality_flag"].eq(0)
    )
    prepared_group["water_vapor_quality_flag"] = np.where(
        ~valid_water, 2, np.where(original_inputs, 0, 1)
    ).astype(np.int8)

    wholly_invalid = []
    for output_name, spec in ENVIRONMENT_SPECS.items():
        if prepared_group[spec["flag"]].eq(2).all():
            wholly_invalid.append(spec["label"])
    invalid_geometry = []
    if prepared_group["chamber_area_quality_flag"].eq(2).any():
        invalid_geometry.append("chamber area")
    if prepared_group["chamber_volume_quality_flag"].eq(2).any():
        invalid_geometry.append("chamber volume")

    flag_columns = [
        "pressure_quality_flag",
        "temperature_quality_flag",
        "relative_humidity_quality_flag",
        "water_vapor_quality_flag",
        "chamber_area_quality_flag",
        "chamber_volume_quality_flag",
    ]
    all_flags = prepared_group[flag_columns].to_numpy(dtype=np.int8)
    if wholly_invalid or invalid_geometry or np.any(all_flags == 2):
        quality_flag = 2
        status = "invalid_auxiliary"
        details = []
        if wholly_invalid:
            details.append("Wholly invalid required variables: " + ", ".join(wholly_invalid))
        if invalid_geometry:
            details.append("Invalid mapped geometry: " + ", ".join(invalid_geometry))
        detail = "; ".join(details) or (
            "At least one required auxiliary value is unresolved or invalid."
        )
    elif np.any(all_flags == 1):
        quality_flag = 1
        status = "usable_with_fills_or_defaults"
        detail = "One or more values were interpolated, edge-filled, or supplied by geometry defaults."
    else:
        quality_flag = 0
        status = "valid_original"
        detail = "All required values are original and valid."

    prepared_group["measurement_quality_flag"] = np.int8(quality_flag)
    prepared_group["measurement_status"] = status
    prepared_group["measurement_status_detail"] = detail
    return prepared_group


def _prepare_batch(args):
    # Top-level so imap can pickle it. One task per batch of whole
    # measurements: pickling one DataFrame per measurement costs more than
    # the work itself.
    batch, ranges = args
    pieces = [
        prepare_measurement(group, ranges)
        for _, group in batch.groupby(GROUP_KEYS, sort=False, dropna=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def prepare_measurements(frame, ranges, processes=None, progress_callback=None):
    '''
    Run prepare_measurement on every chamber/measurement of frame. Returns a
    list of prepared DataFrames, each holding a batch of whole measurements, in
    the order the measurements first appear in frame. The results carry no
    DataFrame.attrs.

    processes: number of worker processes; None uses every CPU. With 1, or
    fewer than MIN_PARALLEL_MEASUREMENTS measurements, the work stays in this
    process.
    progress_callback: called as progress_callback(completed, total, message)
    after each batch, counting measurements.
    '''
    # attrs can list every source file, and pandas deep-copies them on almost
    # every operation and pickles them with every batch.
    frame = frame.copy(deep=False)
    frame.attrs = {}
    measurement_numbers = frame.groupby(GROUP_KEYS, sort=False, dropna=False).ngroup()
    total = int(measurement_numbers.max()) + 1 if len(frame) else 0
    processes = mp.cpu_count() if processes is None else max(int(processes), 1)
    parallel = processes > 1 and total >= MIN_PARALLEL_MEASUREMENTS
    # A few batches per worker balances the load; in this process, about 100
    # batches keep the progress callback moving.
    batch_count = processes * 8 if parallel else 100
    per_batch = max(1, math.ceil(total / batch_count))
    batches = [batch for _, batch in frame.groupby(measurement_numbers // per_batch)]
    sizes = [min(per_batch, total - index * per_batch) for index in range(len(batches))]
    tasks = ((batch, ranges) for batch in batches)
    if not parallel:
        return _collect(map(_prepare_batch, tasks), sizes, total, progress_callback)
    with mp.Pool(min(processes, len(batches))) as pool:
        return _collect(pool.imap(_prepare_batch, tasks), sizes, total, progress_callback)


def _collect(results, sizes, total, progress_callback):
    prepared, completed = [], 0
    for result, size in zip(results, sizes):
        prepared.append(result)
        completed += size
        if progress_callback is not None:
            progress_callback(completed, max(total, 1), f"Prepared {completed}/{total} measurements")
    return prepared
