"""Start the Dask scheduler and local workers on the MacBook."""

import argparse
import asyncio
from ipaddress import ip_address


async def serve(host, port, nworkers=1):
    from distributed import LocalCluster

    async with LocalCluster(
        host=host,
        scheduler_port=port,
        n_workers=nworkers,
        threads_per_worker=1,
        processes=True,
        dashboard_address="127.0.0.1:8787",
        asynchronous=True,
    ) as cluster:
        address = cluster.scheduler_address
        print(f"Notebook scheduler address: {address}", flush=True)
        print(f"On the Mac mini, run: python start_dask_worker.py {address}", flush=True)
        print(f"MacBook workers ready: {nworkers}. Press Ctrl+C to stop.", flush=True)
        await cluster.scheduler.finished()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="MacBook LAN or VPN address (default: detect automatically)")
    parser.add_argument("--port", type=int, default=8786, help="scheduler port (default: 8786)")
    parser.add_argument("--nworkers", type=int, default=1, help="worker processes on the MacBook (default: 1)")
    args = parser.parse_args()
    if args.nworkers < 1:
        parser.error("--nworkers must be at least 1")

    try:
        from distributed.utils import get_ip
    except ImportError:
        parser.exit(1, 'Install Dask in this environment: python -m pip install -e ".[dask]"\n')

    host = args.host
    if host is None:
        try:
            host = get_ip()
        except OSError:
            parser.error("Could not detect a network address. Pass --host with the MacBook's LAN or VPN IP.")
        if ip_address(host).is_loopback:
            parser.error("Detected a loopback address. Pass --host with the MacBook's LAN or VPN IP.")
    try:
        asyncio.run(serve(host, args.port, args.nworkers))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
