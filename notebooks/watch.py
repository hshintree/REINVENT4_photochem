"""Watchdog for the local campaigns. Detects stalls; does not just show progress.

THE GAP THIS CLOSES. campaign.py and dft_ladder.py print a line when a unit
COMPLETES, so a worker stuck inside one calculation produces no output at all and
looks identical to a worker that is merely slow. Both codepaths are bounded (xTB:
400 L-BFGS steps x 300 SCF iterations; pyscf: max_cycle=100), so a true infinite
hang is unlikely -- but "bounded" can still mean an hour, and you cannot tell
without looking.

The cache is the ground truth: every completed unit commits a row with a timestamp
and its wall time. So progress, rate, and stalls are all queryable from outside the
running process, with no instrumentation in it.

STALL RULE: warn when time-since-last-commit exceeds max(3 x the recent median unit
time, 300 s). Three times the median is generous enough not to cry wolf on a slow
molecule, tight enough to notice a wedged worker within a few minutes.

    python notebooks/watch.py                 # one-shot
    python notebooks/watch.py --follow        # poll every 60 s
    python notebooks/watch.py --follow --every 20
"""
from __future__ import annotations
import argparse, datetime as dt, os, sqlite3, subprocess, sys, time

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xtb_cache.sqlite")


def snapshot(db=DB):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    now = dt.datetime.now()
    out = {"now": now}
    out["totals"] = c.execute(
        "SELECT method, status, COUNT(*) FROM calc GROUP BY method, status").fetchall()
    out["windows"] = {}
    for w in (2, 5, 15, 60):
        cut = (now - dt.timedelta(minutes=w)).strftime("%Y-%m-%dT%H:%M:%S")
        out["windows"][w] = c.execute(
            "SELECT COUNT(*) FROM calc WHERE created > ?", (cut,)).fetchone()[0]
    last = c.execute("SELECT created, method, seconds FROM calc "
                     "ORDER BY created DESC LIMIT 1").fetchone()
    out["last"] = last
    if last:
        t = dt.datetime.strptime(last[0], "%Y-%m-%dT%H:%M:%S")
        out["since_last_s"] = (now - t).total_seconds()
    # Threshold must reflect the SLOWEST kind of unit in flight, not a global
    # median. The cache mixes ~30 s xTB optimisations with ~40 min DFT single
    # points; taking one median across both produced a 5-minute threshold and a
    # STALL WARNING on a perfectly healthy DFT job 16 minutes into a 40-minute
    # calculation. A monitor that cries wolf is one you stop reading.
    cut = (now - dt.timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")
    per = {}
    for meth, sec in c.execute(
            "SELECT method, seconds FROM calc WHERE created > ? AND seconds IS NOT NULL",
            (cut,)):
        per.setdefault(meth, []).append(sec)
    meds = []
    for meth, v in per.items():
        v.sort()
        meds.append(v[len(v) // 2])
    out["median_unit_s"] = max(meds) if meds else None
    out["per_method_median"] = {m: sorted(v)[len(v) // 2] for m, v in per.items()}
    c.close()
    return out


def procs():
    try:
        raw = subprocess.run(["ps", "-eo", "pid,pcpu,etime,command"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    keep = []
    for ln in raw.splitlines():
        # only the PYTHON workers; the parent /bin/zsh wrapper always sits at 0% CPU
        # and was being flagged as a deadlock, which is exactly the kind of false
        # alarm that trains you to ignore the monitor.
        p = ln.split(None, 3)
        if len(p) < 4:
            continue
        cmd = p[3]
        # Match on the EXECUTABLE, not anywhere in the line: the parent
        # `/bin/zsh -c "... python campaign.py ..."` wrapper contains "python" too,
        # sits permanently at 0% CPU, and was being reported as a deadlock. A monitor
        # that cries wolf is a monitor you stop reading.
        if not cmd.split()[0].endswith(("python", "python3")):
            continue
        # keep this list in step with the scripts that write to the cache --
        # dft_molecule.py was missing and the monitor reported "no campaign
        # process running" while it was mid-calculation.
        if not any(k in cmd for k in ("campaign.py", "dft_ladder.py",
                                      "triene_ladder.py", "dft_molecule.py",
                                      "closed_manifold_check.py")):
            continue
        if "watch.py" in cmd:
            continue
        script = next((w for w in cmd.split() if w.endswith(".py")), cmd[:40])
        keep.append((p[0], float(p[1]), p[2], os.path.basename(script)))
    return keep


def report(db=DB) -> bool:
    """Print one report. Returns True if everything looks healthy."""
    s = snapshot(db)
    print(f"[{s['now']:%H:%M:%S}]")
    ok_n = sum(n for _m, st, n in s["totals"] if st == "ok")
    bad = [(m, n) for m, st, n in s["totals"] if st != "ok"]
    print(f"  cache        {ok_n} ok" + (f", failures: {bad}" if bad else ", 0 failures"))
    w = s["windows"]
    print(f"  commits      2min {w[2]:4d}   5min {w[5]:4d}   15min {w[15]:4d}   60min {w[60]:4d}")
    if s["median_unit_s"]:
        print(f"  slowest median unit  {s['median_unit_s']:.0f} s "
              f"(per method, last 12 h: "
              + ", ".join(f"{m.split(chr(47))[0][:12]} {v:.0f}s"
                          for m, v in s.get("per_method_median", {}).items()) + ")")

    running = procs()
    if running:
        for pid, cpu, et, cmd in running:
            flag = "  <-- 0% CPU: possible deadlock (see the xtb/OpenMP note in env memory)" \
                   if cpu < 1.0 else ""
            print(f"  pid {pid:<7s} cpu {cpu:6.1f}%  up {et:>9s}  {cmd}{flag}")
    else:
        print("  no campaign process running")

    healthy = True
    if running:
        thresh = max(3 * (s["median_unit_s"] or 60), 300)
        sl = s.get("since_last_s")
        if sl is None:
            print("  STALL?  nothing has ever been committed"); healthy = False
        elif sl > thresh:
            print(f"  ** STALL WARNING ** {sl/60:.1f} min since the last commit, "
                  f"threshold {thresh/60:.1f} min")
            print("     check: is one worker inside a pathological molecule? the job is")
            print("     resumable, so killing and re-running loses at most one unit each.")
            healthy = False
        else:
            print(f"  last commit  {sl:.0f} s ago (stall threshold {thresh/60:.1f} min)  OK")
        if all(cpu < 1.0 for _p, cpu, _e, _c in running):
            print("  ** ALL PROCESSES AT 0% CPU ** — this is the signature of the known"
                  " xtb/OpenMP deadlock; OMP_NUM_THREADS must be 1 for xTB workers")
            healthy = False
    return healthy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--db", default=DB)
    a = ap.parse_args()
    if not a.follow:
        return 0 if report(a.db) else 1
    try:
        while True:
            report(a.db)
            print()
            time.sleep(a.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
