"""Optional Dask runner for whole-measurement fits."""

import io
import logging
import socket
import threading
import traceback
from contextlib import contextmanager, nullcontext
from functools import partial

from ._logging import get_logger

logger = get_logger(__name__)
CONNECTION_TIMEOUT = 10  # seconds for each of connection and worker registration


def _fit_with_log(measurement, *, fit):
    # A worker may run several fits at once. Capture just this task's records,
    # without changing the worker's logger levels or opening the client's path.
    key, _ = measurement
    with io.StringIO() as diagnostics:
        diagnostics.write(f'Processing ID {key} on {socket.gethostname()}\n')
        handler = logging.StreamHandler(diagnostics)
        thread = threading.get_ident()
        handler.addFilter(lambda record: record.thread == thread)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        package_logger = logging.getLogger('soilgasflux_fcs')
        package_logger.addHandler(handler)
        try:
            result = fit(measurement)
            return result, diagnostics.getvalue()
        except Exception as exc:
            # Carry warnings preceding the exception back with the remote traceback.
            exc.soilgasflux_diagnostics = diagnostics.getvalue()
            raise
        finally:
            package_logger.removeHandler(handler)
            handler.close()


def _map_measurements(client, log, fit, measurements):
    if log is None:
        return client.gather(client.map(fit, measurements, pure=False))

    futures = client.map(partial(_fit_with_log, fit=fit), measurements, pure=False)
    results = []
    first_error = None
    for future in futures:
        try:
            result, diagnostics = future.result()
        except Exception as exc:
            if hasattr(exc, 'soilgasflux_diagnostics'):
                log.write(exc.soilgasflux_diagnostics)
            else:
                traceback.print_exc(file=log)
            if first_error is None:
                first_error = exc
        else:
            log.write(diagnostics)
            results.append(result)
        log.flush()
    if first_error is not None:
        raise first_error
    return results


@contextmanager
def dask_mapper(address, log_path):
    try:
        from distributed import Client, TimeoutError as DaskTimeoutError
    except ImportError as exc:
        raise ImportError('Dask batch processing requires the optional extra. Install it '
                          'on both computers with python -m pip install -e ".[dask]".') from exc

    try:
        client = Client(address, timeout=CONNECTION_TIMEOUT, set_as_default=False)
    except (OSError, DaskTimeoutError) as exc:
        raise ConnectionError(f'Could not connect to the Dask scheduler at {address}. '
                              'Check the scheduler address and LAN or VPN connection.') from exc

    with client:
        try:
            client.wait_for_workers(1, timeout=CONNECTION_TIMEOUT)
        except DaskTimeoutError as exc:
            raise RuntimeError(f'No workers registered with {address}. '
                               f'Start a worker with: dask worker {address}') from exc
        logger.info('Connected to Dask scheduler at %s', address)
        log_file = nullcontext() if log_path is None else open(log_path, 'a', encoding='utf-8')
        with log_file as log:
            yield partial(_map_measurements, client, log)
