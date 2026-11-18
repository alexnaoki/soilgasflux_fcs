"""Start a worker on the Mac mini using the scheduler address printed by the MacBook."""

import argparse
import os
import sys
from importlib.util import find_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheduler_address", help="e.g. tcp://192.168.1.10:8786")
    parser.add_argument("--nworkers", type=int, default=1, help="worker processes (default: 1)")
    parser.add_argument("--host", help="Mac mini LAN or VPN address (default: selected by Dask)")
    args = parser.parse_args()
    if args.nworkers < 1:
        parser.error("--nworkers must be at least 1")
    if find_spec("distributed") is None or find_spec("dask") is None:
        parser.exit(1, 'Install Dask in this environment: python -m pip install -e ".[dask]"\n')

    # Run Dask's worker CLI in this interpreter so it handles startup and Ctrl+C.
    command = [sys.executable, "-m", "dask", "worker", args.scheduler_address,
               "--nworkers", str(args.nworkers), "--nthreads", "1", "--death-timeout", "30s"]
    if args.host:
        command.extend(["--host", args.host])
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
