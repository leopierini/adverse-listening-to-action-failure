#!/usr/bin/env python
"""Live cost meter for the thesis GPU VM.

WHY NOT JUST USE THE BILLING CONSOLE
------------------------------------
GCP's billing reports lag by hours (up to ~24h for finalised data), and budget
alerts fire off that same delayed feed. Neither is usable for "how much is this
session costing me right now".

Compute Engine bills the VM per second while it is RUNNING, so uptime x rate is
exact and has no lag. This reads the instance's real state and its
`lastStartTimestamp` straight from the API and does that arithmetic.

The rate is an ESTIMATE from published on-demand pricing (tracker §14 records
~EUR 3.60-3.90/hr for a2-highgpu-1g in europe-west4). Override with --rate.
Sustained-use discounts, committed-use discounts and any academic credits are
NOT modelled, so the number reads HIGH — which is the safe direction for a
budget ceiling.

Usage:
    ./venv/bin/python scripts/gcp_cost_meter.py
    ./venv/bin/python scripts/gcp_cost_meter.py --watch 60
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

ZONE = "europe-west4-a"
INSTANCE = "instance-pierini-gpu"

# a2-highgpu-1g (12 vCPU, 85 GB, 1x A100 40GB), europe-west4, on-demand.
# Central value of the EUR 3.60-3.90/hr range recorded in tracker §14.
DEFAULT_RATE_EUR_HR = 3.75

# 500 GB pd-standard in europe-west4. Billed continuously, VM running or not.
DISK_GB = 500
DISK_EUR_GB_MONTH = 0.044

BUDGET_EUR = 500.0


def gcloud_json(args):
    p = subprocess.run(["gcloud"] + args + ["--format=json"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if p.returncode != 0:
        raise SystemExit("gcloud failed: " + p.stderr.decode("utf-8", "replace")[:400])
    return json.loads(p.stdout.decode("utf-8", "replace"))


def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "{}h {:02d}m {:02d}s".format(h, m, s)


def snapshot(rate):
    inst = gcloud_json(["compute", "instances", "describe", INSTANCE,
                        "--zone", ZONE])
    status = inst.get("status")
    started = inst.get("lastStartTimestamp")
    stopped = inst.get("lastStopTimestamp")

    out = {"status": status, "started": started, "stopped": stopped,
           "uptime_s": 0.0, "compute_eur": 0.0}

    if status == "RUNNING" and started:
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        out["uptime_s"] = (datetime.now(timezone.utc) - t0).total_seconds()
        out["compute_eur"] = out["uptime_s"] / 3600.0 * rate

    out["disk_eur_month"] = DISK_GB * DISK_EUR_GB_MONTH
    out["disk_eur_day"] = out["disk_eur_month"] / 30.0
    return out


def render(s, rate):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print("=" * 60)
    print("  {}   {}".format(INSTANCE, now))
    print("=" * 60)
    print("  stato          : {}".format(s["status"]))

    if s["status"] == "RUNNING":
        print("  acceso da      : {}".format(s["started"]))
        print("  tempo acceso   : {}".format(fmt_dur(s["uptime_s"])))
        print("  tariffa        : EUR {:.2f}/ora (stima on-demand)".format(rate))
        print("  --------------------------------------------------------")
        print("  COSTO SESSIONE : EUR {:.2f}".format(s["compute_eur"]))
        print("  --------------------------------------------------------")
        print("  proiezioni     : 1h = EUR {:.2f} | 4h = EUR {:.2f} | 8h = EUR {:.2f}"
              .format(rate, rate * 4, rate * 8))
        print("\n  >>> LA MACCHINA STA FATTURANDO. Spegnila quando hai finito:")
        print("      gcloud compute instances stop {} --zone={}".format(INSTANCE, ZONE))
    else:
        print("  compute        : EUR 0.00 — non sta fatturando")
        if s["stopped"]:
            print("  ultimo stop    : {}".format(s["stopped"]))

    print("\n  disco 500 GB pd-standard (si paga SEMPRE, anche da spenta):")
    print("      ~EUR {:.2f}/mese  (~EUR {:.2f}/giorno)".format(
        s["disk_eur_month"], s["disk_eur_day"]))
    print("      = {:.1f}% del budget di EUR {:.0f} ogni mese che passa".format(
        s["disk_eur_month"] / BUDGET_EUR * 100, BUDGET_EUR))
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Live cost meter for the GPU VM.")
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE_EUR_HR,
                    help="EUR/hour for the running VM (default {})".format(
                        DEFAULT_RATE_EUR_HR))
    ap.add_argument("--watch", type=int, metavar="SEC", default=0,
                    help="refresh every SEC seconds (Ctrl-C to stop)")
    args = ap.parse_args()

    try:
        while True:
            render(snapshot(args.rate), args.rate)
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nmisuratore fermato (la VM NON e' stata toccata)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
