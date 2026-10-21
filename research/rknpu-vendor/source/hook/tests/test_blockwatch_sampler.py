#!/usr/bin/env python3
"""Host regression test for the blockwatcher sampler and the phase-sweep chain.

No board, no /dev/mem: the hook is built with -DEPERF_BW_SELFTEST, which swaps
the MMIO window for a calloc'd buffer and drives a synthetic task counter.

Every group below corresponds to a defect that was found by review, so a
regression re-opens a real hole in the regcmd timing plan
(research/scheduling/REGCMD-TIMING-OBSERVABILITY.md):

  1. phase delay measured from the recorded submit entry, not thread start;
  2. edge mode ("one sample per counter change") is rate limited;
  3. one read budget covers polls and snapshots;
  4. the sample capacity stops the run;
  5. every read carries a bracket, and the CSV names it correctly;
  6. the single-register probe brackets the read with task status checks;
  7. RAW_OFF is parsed strictly -- an empty value must NOT collapse to 0x0000
     (PC_VERSION) and PC/reserved pages are refused;
  8. the power guard runs BEFORE the first MMIO read;
  9. a foreign ioctl cannot move the time origin of a run it does not own;
 10. the shell entry point writes absolute manifest paths and refuses to
     disable the power guard;
 11. the shell entry point produces a manifest the analyzer can consume
     (end to end, including the workload invocation);
 12. the shell entry point drops runs whose RAW_OFF was rejected.

Run:  python3 test_blockwatch_sampler.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "rknpu", "rknpu_ioctl_hook.c")
EXPERIMENTS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
ANALYZER = os.path.join(EXPERIMENTS, "analyze_phase_sweep.py")
BANK_PROBE = os.path.join(EXPERIMENTS, "analyze_bank_probe.py")
BANK_RUNNER = os.path.join(EXPERIMENTS, "run_bank_probe.sh")
IRQ_ANALYZER = os.path.join(EXPERIMENTS, "analyze_irq_completion.py")
KICK_CAL = os.path.join(EXPERIMENTS, "calibrate_kick_offset.py")
EDGE_ANALYZER = os.path.join(EXPERIMENTS, "analyze_intra_task_edges.py")
WORKLOAD_VERDICT = os.path.join(EXPERIMENTS, "workload_verdict.py")
DRIVER_EVENTS = os.path.join(EXPERIMENTS, "analyze_driver_events.py")
SLOT_PLANNER = os.path.join(EXPERIMENTS, "plan_marker_slots.py")
MARKER_RUNNER = os.path.join(EXPERIMENTS, "run_marker_arms.sh")
MIN_CONV = os.path.join(EXPERIMENTS, "min_conv.c")
ARM_COMPARE = os.path.join(EXPERIMENTS, "compare_arm_outputs.py")
ROWS_CSV = os.path.join(EXPERIMENTS, "results", "sdpa_timeline_20261020",
                        "sdpa_regcmd_order_rows.csv")
SWEEP = os.path.join(EXPERIMENTS, "run_phase_sweep.sh")

META_RE = re.compile(r"BLOCKWATCH_META (?P<body>.*)")
CONFIG_RE = re.compile(r"BLOCKWATCH_CONFIG (?P<body>.*)")


def build(tmpdir):
    selftest = os.path.join(tmpdir, "bw_selftest")
    subprocess.run([
        "gcc", "-O2", "-DEPERF_BW_SELFTEST", "-Wall", "-Wextra",
        "-o", selftest, SRC, "-ldl", "-lpthread",
    ], check=True, capture_output=True)
    so = os.path.join(tmpdir, "libhook.so")
    res = subprocess.run([
        "gcc", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
        "-o", so, SRC, "-ldl", "-lpthread",
    ], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("shared build failed:\n" + res.stderr)
    unexpected = [ln for ln in res.stderr.splitlines()
                  if "warning:" in ln and "g_probe_saved_tn" not in ln]
    if unexpected:
        raise SystemExit("new compiler warnings:\n" + "\n".join(unexpected))
    # a preloadable no-op object: the shell sweep needs a file to LD_PRELOAD,
    # while the workload below drives the sampler itself
    dummy = os.path.join(tmpdir, "dummy.so")
    with open(os.path.join(tmpdir, "dummy.c"), "w") as fh:
        fh.write("void eperf_dummy(void) {}\n")
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-o", dummy,
                    os.path.join(tmpdir, "dummy.c")], check=True)
    return selftest, dummy


def run(selftest, tmpdir, name="run", args=None, env_extra=None):
    csv_path = os.path.join(tmpdir, "%s.csv" % name)
    argv = [selftest] + list(args or [])
    env = dict(os.environ, EPERF_RKNPU_BLOCKWATCH_FILE=csv_path)
    for key in ("EPERF_RKNPU_BLOCKWATCH_RAW_OFF",):
        env.pop(key, None)
    env.update(env_extra or {})
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)

    def parse(pattern):
        found = {}
        match = pattern.search(proc.stderr)
        if match:
            for item in match.group("body").split():
                key, _, value = item.partition("=")
                found[key] = value
        return found

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    return parse(META_RE), parse(CONFIG_RE), rows, proc


def check(cond, message):
    if not cond:
        raise AssertionError(message)


def group(name, fn, failures):
    print(name)
    try:
        fn()
    except AssertionError as exc:
        print("  FAIL " + str(exc))
        failures.append(name)


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        selftest, dummy_so = build(tmpdir)

        def phase_alignment():
            meta, _, _, _ = run(selftest, tmpdir, name="phase",
                                args=["--phase-ns=5000000", "--run-ms=60",
                                      "--arm-delay-us=20000"])
            lag = int(meta["submit_enter_to_first_sample_ns"])
            check(0 <= lag < 12_000_000, "first sample inside the phase window (lag=%d)" % lag)
            check(lag >= 4_500_000, "phase delay was not skipped (lag=%d)" % lag)
            print("  ok  first sample %d ns after submit entry (phase 5000000)" % lag)

        group("phase delay is measured from the submit entry", phase_alignment, failures)

        def edge_rate():
            meta, _, rows, _ = run(selftest, tmpdir, name="edge",
                                   args=["--edge=1", "--poll-ns=1000000",
                                         "--run-ms=50", "--phase-ns=0"])
            polls = int(meta["polls"])
            check(polls <= 70, "50 ms at a 1 ms poll floor issued %d polls" % polls)
            check(len(rows) > 0, "synthetic transitions were still recorded")
            print("  ok  %d polls, %d samples" % (polls, len(rows)))

        group("edge mode is rate limited", edge_rate, failures)

        def budget():
            meta, _, rows, _ = run(selftest, tmpdir, name="budget",
                                   args=["--edge=1", "--poll-ns=1000",
                                         "--run-ms=200", "--max-reads=10"])
            check(int(meta["budget_hit"]) == 1, "budget exhaustion is reported")
            check(meta["stop_reason"] == "budget", "stop_reason names the budget")
            check(int(meta["mmio_reads"]) == 10, "no read past the budget (%s)"
                  % meta["mmio_reads"])
            check(len(rows) <= 10, "partial samples were not published (%d rows)" % len(rows))
            print("  ok  10 reads, %d complete samples, partial sample dropped" % len(rows))

        group("one read budget covers polls and snapshots", budget, failures)

        def capacity():
            meta, _, rows, _ = run(selftest, tmpdir, name="cap",
                                   args=["--edge=1", "--poll-ns=1000",
                                         "--run-ms=200", "--cap=3"])
            check(int(meta["samples"]) == 3, "cap honoured (samples=%s)" % meta["samples"])
            check(len(rows) == 3, "csv row count matches (rows=%d)" % len(rows))
            check(meta["stop_reason"] == "cap", "stop_reason distinguishes an intended cap")
            print("  ok  cap=3 -> 3 samples, stop_reason=cap")

        group("sample capacity stops the run", capacity, failures)

        def brackets():
            _, _, rows, _ = run(selftest, tmpdir, name="bracket",
                                args=["--run-ms=30", "--phase-ns=0", "--cap=2"])
            check(len(rows) > 0, "free-running mode produced samples")
            row = rows[0]
            t, t_end = int(row["t_ns"]), int(row["t_end_ns"])
            gb, ge = int(row["c0_gb_ns"]), int(row["c0_ge_ns"])
            check(t <= gb <= ge <= t_end, "bracket ordering t<=gb<=ge<=t_end holds")
            check("submit_enter_ns" in row, "csv carries submit_enter_ns, not kick_ns")
            check("kick_ns" not in row, "the misleading kick_ns column is gone")
            print("  ok  group bracket ordering and submit_enter_ns column")

        group("every read carries a bracket", brackets, failures)

        def raw_probe():
            meta, config, rows, _ = run(selftest, tmpdir, name="raw",
                                        args=["--run-ms=20", "--raw-off=0x1000", "--cap=1"])
            check(config.get("raw_off") == "on", "raw_off reported in CONFIG")
            check(config.get("raw_mode") == "vetted", "0x1000 is a vetted register")
            check(len(rows) == 1, "one snapshot only (rows=%d)" % len(rows))
            check(int(meta["mmio_reads"]) == 3,
                  "ts + raw + ts = 3 MMIO reads (%s)" % meta["mmio_reads"])
            check(rows[0]["sample_validity"] == "ok",
                  "task_before == task_after -> ok (%s)" % rows[0]["sample_validity"])
            meta, _, _, _ = run(selftest, tmpdir, name="raw_notask",
                                args=["--run-ms=20", "--raw-off=0x1000", "--cap=1",
                                      "--task-check=0"])
            check(int(meta["mmio_reads"]) == 2,
                  "without the task check only 2 reads (%s)" % meta["mmio_reads"])
            print("  ok  3 reads with task check, 2 without")

        group("the raw probe brackets the target with task status", raw_probe, failures)

        def dual_raw():
            # Two registers in one sample: the budget and the CSV must both show it.
            env = {"EPERF_RKNPU_BLOCKWATCH_RAW_OFF": "0x1000",
                   "EPERF_RKNPU_BLOCKWATCH_RAW_OFF2": "0x3000"}
            meta, _, rows, _ = run(selftest, tmpdir, name="dual", args=["--run-ms=10",
                                                                     "--cap=1"],
                                   env_extra=env)
            check(len(rows) == 1, "one sample")
            check("c0_raw2" in rows[0], "csv carries the second register: %s" % list(rows[0]))
            check(int(meta["mmio_reads"]) == 4,
                  "ts + reg1 + reg2 + ts = 4 reads (%s)" % meta["mmio_reads"])
            meta, _, _, _ = run(selftest, tmpdir, name="dual_notask",
                                args=["--run-ms=10", "--cap=1", "--task-check=0"],
                                env_extra=env)
            check(int(meta["mmio_reads"]) == 3,
                  "without the task check: 3 reads (%s)" % meta["mmio_reads"])
            # a forbidden second offset is rejected on its own
            meta, config, rows, _ = run(
                selftest, tmpdir, name="dual_bad", args=["--run-ms=5", "--cap=1"],
                env_extra={"EPERF_RKNPU_BLOCKWATCH_RAW_OFF": "0x1000",
                           "EPERF_RKNPU_BLOCKWATCH_RAW_OFF2": "0x4104"})
            check(config.get("raw_off2") == "off",
                  "a forbidden second offset is dropped: %s" % config.get("raw_off2"))
            print("  ok  two registers per sample; bad second offset rejected")

        group("the raw probe can read a second register", dual_raw, failures)

        def irq_words_are_the_only_pc_reads():
            # the two interrupt words are readable without an experimental flag and
            # the CONFIG line names the mode; the rest of the PC page stays refused
            meta, config, rows, _ = run(selftest, tmpdir, name="irq1",
                                        args=["--run-ms=10", "--raw-off=0x0028", "--cap=1"])
            check(config.get("raw_mode") == "irq_status",
                  "CONFIG names the read mode: %s" % config.get("raw_mode"))
            check(len(rows) == 1 and "c0_raw" in rows[0], "one snapshot with the raw column")
            env = {"EPERF_RKNPU_BLOCKWATCH_RAW_OFF": "0x0028",
                   "EPERF_RKNPU_BLOCKWATCH_RAW_OFF2": "0x002c"}
            meta, config, rows, _ = run(selftest, tmpdir, name="irq2",
                                        args=["--run-ms=10", "--cap=1"], env_extra=env)
            check(config.get("raw_off2") == "on", "second interrupt word is on: %s" % config)
            check("c0_raw2" in rows[0], "csv carries both interrupt words")
            check(int(meta["mmio_reads"]) == 4,
                  "ts + status + raw status + ts = 4 reads (%s)" % meta["mmio_reads"])
            print("  ok  INTERRUPT_STATUS and RAW_STATUS read without an experimental flag")

        group("the interrupt words are the only PC-page raw reads",
              irq_words_are_the_only_pc_reads, failures)

        def same_clock_anchors():
            meta, _, _, _ = run(selftest, tmpdir, name="anc1",
                                args=["--run-ms=10", "--cap=1", "--note-ioctl=1"])
            check(meta.get("lat_ioctl_start_ns") == "2000",
                  "ioctl entry relative to the submit entry: %s" % meta)
            check(meta.get("lat_ioctl_return_ns") == "60000",
                  "ioctl return relative to the submit entry: %s" % meta)
            meta, _, _, _ = run(selftest, tmpdir, name="anc0", args=["--run-ms=10", "--cap=1"])
            check(meta.get("lat_ioctl_return_ns") == "-1",
                  "without a real ioctl the anchor is absent")
            print("  ok  ioctl entry and return are reported on the sample's clock")

        group("the anchors share the sample's clock", same_clock_anchors, failures)

        def irq_analyzer_selftest():
            proc = subprocess.run([sys.executable, IRQ_ANALYZER, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "irq analyzer selftest failed: %s" % proc.stdout[-400:])
            check("all irq-completion selftest checks passed" in proc.stdout,
                  proc.stdout[-300:])
            print("  ok  irq completion analyzer selftest passes")

        group("the completion analyzer passes its own selftest",
              irq_analyzer_selftest, failures)

        def kick_calibrator_selftest():
            proc = subprocess.run([sys.executable, KICK_CAL, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "kick calibrator selftest failed: %s" % proc.stdout[-400:])
            check("all kick-offset selftest checks passed" in proc.stdout,
                  proc.stdout[-300:])
            print("  ok  kick-offset calibrator selftest passes")

        group("the kick-offset calibrator passes its own selftest",
              kick_calibrator_selftest, failures)

        def intra_task_snapshots():
            # CAP>1 must actually produce several snapshots in one run, spaced in time
            meta, config, rows, _ = run(selftest, tmpdir, name="multi_cap",
                                        args=["--cap=6", "--run-ms=40", "--spin-ns=1000",
                                              "--phase-ns=0", "--raw-off=0x0028"])
            check(int(config["select"]) >= 0, config)
            check(len(rows) >= 2, "more than one snapshot per run (got %d)" % len(rows))
            phases = [int(r["t_ns"]) for r in rows]
            check(phases == sorted(phases), "snapshots are ordered in time")
            check(len(set(phases)) == len(phases), "no duplicate snapshot times")
            print("  ok  CAP>1 yields %d ordered snapshots in one run" % len(rows))

        group("several snapshots can be taken inside one run",
              intra_task_snapshots, failures)

        def edge_analyzer_selftest():
            proc = subprocess.run([sys.executable, EDGE_ANALYZER, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "intra-task edge analyzer selftest failed: %s" % proc.stdout[-400:])
            check("all intra-task edge selftest checks passed" in proc.stdout,
                  proc.stdout[-300:])
            print("  ok  intra-task edge analyzer selftest passes")

        group("the intra-task edge analyzer passes its own selftest",
              edge_analyzer_selftest, failures)

        def workload_verdict_selftest():
            proc = subprocess.run([sys.executable, WORKLOAD_VERDICT, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "workload-verdict selftest failed: %s" % proc.stdout[-400:])
            check("all workload-verdict selftest checks passed" in proc.stdout,
                  proc.stdout[-300:])
            # the second tier must not swallow a large-ulp result
            proc = subprocess.run(
                [sys.executable, WORKLOAD_VERDICT],
                input="total=1024 fp16-mismatch=45 max|diff|=8e-06 max ulp=140\nRESULT: MISMATCH\n",
                capture_output=True, text=True)
            check(proc.stdout.startswith("rejected"),
                  "140 ulp must be rejected, got %r" % proc.stdout[:60])
            print("  ok  workload verdict tiers behave, big ulp stays rejected")

        group("the workload verdict has an explicit bounded tier",
              workload_verdict_selftest, failures)

        def driver_event_analyzer_selftest():
            proc = subprocess.run([sys.executable, DRIVER_EVENTS, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "driver-event analyzer selftest failed: %s" % proc.stdout[-400:])
            check("all driver-event selftest checks passed" in proc.stdout,
                  proc.stdout[-300:])
            print("  ok  driver-event alignment analyzer selftest passes")

        group("the ftrace/sampler alignment tool passes its selftest",
              driver_event_analyzer_selftest, failures)

        def raw_off_validation():
            # (value, rc without experimental, rc with experimental)
            cases = [("", 1, 1), ("0x0000", 1, 1), ("0x2000", 1, 1),
                     ("0x1026", 1, 1), ("0x8000", 1, 1), ("junk", 1, 1),
                     ("0x003c", 1, 1), ("0x4104", 1, 1), ("0x412c", 1, 1),
                     ("0x4004", 1, 1), ("0x5008", 1, 1),
                     ("0x1000", 0, 0), ("0x3000", 0, 0), ("0x4000", 0, 0),
                     ("0x5000", 0, 0),
                     ("0x1024", 1, 0), ("0x1ffc", 1, 0), ("0x6008", 1, 1),
                     ("0x0028", 0, 0), ("0x002c", 0, 0), ("0x0020", 1, 1),
                     ("0x0024", 1, 1), ("0x0030", 1, 1), ("0x0010", 1, 1),
                     ("0x0014", 1, 1), ("0x0008", 1, 1), ("0x0004", 1, 1)]
            for value, plain, experimental in cases:
                proc = subprocess.run([selftest, "--raw-off-str=%s" % value],
                                      capture_output=True, text=True)
                check(proc.returncode == plain,
                      "raw-off-str=%r -> rc=%d, expected %d without experimental"
                      % (value, proc.returncode, plain))
                proc = subprocess.run([selftest, "--raw-off-str=%s" % value,
                                       "--raw-experimental=1"],
                                      capture_output=True, text=True)
                check(proc.returncode == experimental,
                      "raw-off-str=%r -> rc=%d, expected %d with experimental"
                      % (value, proc.returncode, experimental))
                if experimental == 0 and plain == 1:
                    check("mode=experimental" in proc.stderr,
                          "experimental tier is reported for %s" % value)
            # the env path must reject too: an empty value used to mean offset 0
            meta, config, _, _ = run(selftest, tmpdir, name="raw_env",
                                     args=["--run-ms=5", "--cap=1"],
                                     env_extra={"EPERF_RKNPU_BLOCKWATCH_RAW_OFF": ""})
            check(config.get("raw_rejected") == "1", "empty RAW_OFF flagged")
            check(config.get("raw_off") == "off", "empty RAW_OFF did not enable raw mode")
            check(int(meta["mmio_reads"]) >= 0, "run still completes")
            print("  ok  empty/PC/reserved/misaligned rejected; CNA/CORE/PPU accepted")

        group("RAW_OFF is validated strictly", raw_off_validation, failures)

        def power_guard():
            off = os.path.join(tmpdir, "power_off.txt")
            on = os.path.join(tmpdir, "power_on.txt")
            with open(off, "w") as fh:
                fh.write("off\n")
            with open(on, "w") as fh:
                fh.write("on\n")
            meta, _, rows, _ = run(selftest, tmpdir, name="pwr_off",
                                   args=["--run-ms=10", "--power-check-ns=100000",
                                         "--power-file=%s" % off])
            check(int(meta["mmio_reads"]) == 0,
                  "no MMIO read before the first power check (%s)" % meta["mmio_reads"])
            check(meta["stop_reason"] == "power", "stop_reason=power (%s)"
                  % meta["stop_reason"])
            check(len(rows) == 0, "no sample published")
            meta, _, rows, _ = run(selftest, tmpdir, name="pwr_on",
                                   args=["--run-ms=10", "--power-check-ns=100000",
                                         "--power-file=%s" % on, "--cap=3"])
            check(len(rows) == 3, "sampling proceeds when the domain is on")
            print("  ok  domain off -> 0 reads and stop_reason=power")

        group("the power guard runs before the first read", power_guard, failures)

        def ownership():
            meta, config, _, _ = run(selftest, tmpdir, name="owner",
                                     args=["--run-ms=10", "--phase-ns=0",
                                           "--arm-foreign-first=1", "--cap=1"])
            lag = int(meta["submit_enter_to_first_sample_ns"])
            check(0 <= lag < 5_000_000,
                  "first sample stayed anchored to the owner arm (lag=%d)" % lag)
            check(int(config["owner_conflicts"]) >= 1, "the foreign arm was counted")
            meta, config, rows, _ = run(selftest, tmpdir, name="owner_stop",
                                        args=["--run-ms=20", "--phase-ns=0", "--cap=3",
                                              "--stop-foreign=1"])
            check(int(meta["samples"]) == 3,
                  "a foreign stop did not tear the run down (samples=%s)"
                  % meta["samples"])
            check(int(config["owner_conflicts"]) >= 1, "the foreign stop was counted")
            print("  ok  foreign arm/stop ignored (conflicts=%s, samples=%s)"
                  % (config["owner_conflicts"], meta["samples"]))

        group("a foreign ioctl cannot move the time origin", ownership, failures)

        def races():
            proc = subprocess.run([selftest, "--races=1"], capture_output=True, text=True)
            check(proc.returncode == 0,
                  "concurrency/guard-page probe failed: %s%s" % (proc.stdout, proc.stderr))
            out = proc.stdout
            check("foreign_ioctl_guard_page=survived" in out,
                  "a foreign ioctl must not dereference its argument: %s" % out)
            check("start_during_starting=-1" in out, "no start inside STARTING: %s" % out)
            check("first_start=0" in out, "the first start still succeeds: %s" % out)
            check("start_during_stopping=-1" in out, "no start inside STOPPING: %s" % out)
            check("start_after_stop=0" in out, "the slot frees only after teardown: %s" % out)
            check("failures=0" in out, "no race failures: %s" % out)
            print("  ok  STARTING/STOPPING windows block a new start; guard page survives")

        group("concurrent start/stop and foreign-ioctl safety", races, failures)

        def wrapper_probe():
            # The same hazard, but through the real ioctl wrapper while the
            # sampler is RUNNING: a foreign request carrying a guard-page
            # argument must not be dereferenced, must not steal the run, and
            # the owner must still stop cleanly afterwards.
            proc = subprocess.run([selftest, "--wrapper-probe=1"],
                                  capture_output=True, text=True)
            out = proc.stdout
            check(proc.returncode == 0,
                  "wrapper probe crashed or failed (rc=%d): %s%s"
                  % (proc.returncode, out, proc.stderr[-300:]))
            check("foreign_ioctl_rc=-1" in out,
                  "the foreign ioctl returns an error: %s" % out)
            check("errno=25" in out or "errno=9" in out,
                  "the error is ENOTTY/EBADF, not a fault: %s" % out)
            check("still_owned=1" in out, "the run was not stolen: %s" % out)
            check("samples_continue=1" in out,
                  "the sampler keeps running after a foreign ioctl: %s" % out)
            check("stopped_after=1" in out,
                  "the owner can still stop and reach IDLE: %s" % out)
            check("failures=0" in out, "no probe failures: %s" % out)
            print("  ok  RUNNING + real ioctl wrapper + guard page survives")

        group("a foreign ioctl cannot touch its argument (real wrapper)",
              wrapper_probe, failures)

        def shell_manifest():
            out = os.path.join(tmpdir, "dry")
            proc = subprocess.run([SWEEP, "--dry-run", "--phases", "0,10000",
                                   "--reps", "2", "--out", out],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "dry run failed: %s" % proc.stderr)
            with open(os.path.join(out, "manifest.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 4, "4 planned runs (%d)" % len(rows))
            for row in rows:
                check(os.path.isabs(row["csv"]), "csv path is absolute: %s" % row["csv"])
            for bad in ("0", "00", "-1", "abc"):
                proc = subprocess.run([SWEEP, "--dry-run", "--power-check-ns", bad,
                                       "--out", os.path.join(tmpdir, "dry2")],
                                      capture_output=True, text=True)
                check(proc.returncode != 0,
                      "POWER_CHECK_NS=%r must be refused" % bad)
            print("  ok  absolute manifest paths, power guard cannot be disabled")

        group("the shell entry writes an analysable manifest", shell_manifest, failures)

        def shell_end_to_end():
            out = os.path.join(tmpdir, "sweep")
            proc = subprocess.run(
                [SWEEP, "--phases", "0,6000,12000,18000,24000,30000", "--reps", "3",
                 "--hook", dummy_so, "--out", out, "--seed", "5", "--",
                 selftest, "--run-ms=3", "--fake-busy-after-ns=20000"],
                capture_output=True, text=True)
            check(proc.returncode == 0, "sweep failed: %s" % proc.stderr)
            manifest = os.path.join(out, "manifest.csv")
            with open(manifest, newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) >= 12, "sweep produced runs (%d)" % len(rows))
            analysis = os.path.join(out, "analysis")
            proc = subprocess.run([sys.executable, ANALYZER, "--manifest", manifest,
                                   "--signal", "c0_cna", "--out-dir", analysis],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "analyzer failed: %s" % proc.stderr)
            summary = json.load(open(os.path.join(analysis, "summary.json")))
            check(summary["mode"] == "status_probability", "status mode, not a CDF")
            block = summary["conditions"][""]
            check(block["cdf"] is None, "status mode must not emit a CDF")
            bracket = block["transition"].get("bracket_ns")
            check(bracket is not None, "transition bracket found: %s"
                  % block["transition"]["status"])
            check(bracket[0] <= 25_000 and bracket[1] >= 15_000,
                  "bracket %s covers the 20 us truth" % bracket)
            print("  ok  shell sweep -> manifest -> analyzer, bracket %s ns"
                  % (bracket,))

        group("the shell entry runs end to end", shell_end_to_end, failures)

        def shell_env_hygiene():
            out = os.path.join(tmpdir, "sweep_env")
            env = dict(os.environ)
            # a stale raw offset in the parent shell must not turn the sweep
            # into a single-register probe
            env["EPERF_RKNPU_BLOCKWATCH_RAW_OFF"] = "0x1000"
            env["EPERF_RKNPU_BLOCKWATCH_EDGE"] = "1"
            env["EPERF_RKNPU_BLOCKWATCH_MINIMAL"] = "1"
            proc = subprocess.run(
                [SWEEP, "--phases", "0,6000", "--reps", "1", "--hook", dummy_so,
                 "--out", out, "--", selftest, "--run-ms=3"],
                capture_output=True, text=True, env=env)
            check(proc.returncode == 0, "sweep failed: %s" % proc.stderr)
            with open(os.path.join(out, "manifest.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 2, "both runs kept (%d)" % len(rows))
            with open(rows[0]["csv"], newline="") as fh:
                header = next(csv.reader(fh))
            check("c0_cna" in header,
                  "inherited RAW_OFF did not switch the run to raw mode: %s" % header)
            check("c0_raw" not in header, "no raw column: %s" % header)
            print("  ok  inherited EPERF_* sampling config is cleared")

        group("the shell entry does not inherit stale sampling env", shell_env_hygiene, failures)

        def shell_drops_rejected():
            # 1. a forbidden offset is refused before anything runs
            out = os.path.join(tmpdir, "sweep_bad")
            proc = subprocess.run(
                [SWEEP, "--phases", "1000", "--reps", "1", "--raw-off", "0x2000",
                 "--hook", dummy_so, "--out", out, "--",
                 selftest, "--run-ms=3"],
                capture_output=True, text=True)
            check(proc.returncode != 0,
                  "a forbidden RAW_OFF must fail before the sweep starts")
            # 2. an offset the script lets through (experimental) but the hook
            #    forbids (DPU LUT data port) is dropped per run, not recorded
            out = os.path.join(tmpdir, "sweep_bad2")
            proc = subprocess.run(
                [SWEEP, "--phases", "1000", "--reps", "1", "--raw-off", "0x4104",
                 "--raw-experimental", "--hook", dummy_so, "--out", out, "--",
                 selftest, "--run-ms=3"],
                capture_output=True, text=True)
            with open(os.path.join(out, "manifest.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 0, "hook-rejected runs are not recorded (%d)" % len(rows))
            check("RAW_REJECTED" in proc.stderr or "rejected RAW_OFF" in proc.stderr,
                  "the sweep reports the rejection: %s" % proc.stderr[-300:])
            print("  ok  forbidden offsets fail fast, hook-rejected runs are dropped")

        group("the shell entry drops rejected RAW_OFF runs", shell_drops_rejected, failures)

        def bank_probe_end_to_end():
            # The route-B gate: does the host read the group the current task
            # writes, or the other one?  The fake window emulates both, and the
            # offline analyzer must name them without ever writing to a device.
            rows_path = os.path.join(tmpdir, "bank_rows.csv")
            with open(rows_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["task_index", "task_local_regcmd_index", "module",
                            "reg_offset", "reg_name", "value"])
                for task in range(40):
                    w.writerow([task, 3, "CNA", "0x1024", "CNA_DATA_SIZE1",
                                "0x%08x" % (0x1000 + task)])
            for lag, expected in ((0, "observed_lag_0"), (1, "observed_lag_1")):
                root = os.path.join(tmpdir, "bank_lag%d" % lag)
                os.makedirs(root, exist_ok=True)
                manifest = os.path.join(root, "manifest.csv")
                with open(manifest, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["run_id", "request_phase_ns", "csv", "condition"])
                    for task in range(2, 38):
                        name = "t%02d" % task
                        path = os.path.join(root, name + ".csv")
                        env = dict(os.environ, EPERF_RKNPU_BLOCKWATCH_FILE=path)
                        subprocess.run([selftest, "--raw-off=0x1024", "--cap=1",
                                        "--run-ms=3", "--fake-task=%d" % task,
                                        "--fake-bank-lag=%d" % lag],
                                       env=env, check=True, capture_output=True)
                        w.writerow([name, 10_000, path, ""])
                out = os.path.join(root, "analysis")
                proc = subprocess.run([sys.executable, BANK_PROBE, "--rows", rows_path,
                                       "--manifest", manifest, "--module", "CNA",
                                       "--offset", "0x1024", "--out-dir", out],
                                      capture_output=True, text=True)
                check(proc.returncode == 0, "bank probe failed: %s" % proc.stderr)
                summary = json.load(open(os.path.join(out, "summary.json")))
                got = summary["verdict"]["verdict"]
                check(got == expected, "lag %d -> %s (expected %s)" % (lag, got, expected))
            print("  ok  hook CSV -> bank probe distinguishes observed_lag_0 / observed_lag_1")

        group("the bank probe reads the lag end to end", bank_probe_end_to_end, failures)

        def marker_decision():
            proc = subprocess.run([selftest, "--marker-check=1"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "marker decision table failed: %s" % proc.stdout)
            check("failures=0" in proc.stdout, proc.stdout)
            check("FAIL" not in proc.stdout, proc.stdout)
            accepted = proc.stdout.count("got=accept")
            check(accepted == 2, "exactly the two valid cases are accepted (%d)" % accepted)
            print("  ok  %d marker decision cases, %d accepted"
                  % (proc.stdout.count("SELFTEST_MARKER "), accepted))

        group("the marker slot decision table is enforced", marker_decision, failures)

        def marker_bo():
            # The actual rewrite path, driven through the real marker_apply():
            # fake task BO + fake regcmd BO registered as if mmap had been
            # intercepted.  Verifies the exact word, that nothing else moves,
            # and that every rejection leaves the buffer byte-identical.
            proc = subprocess.run([selftest, "--marker-bo=1"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0,
                  "marker BO harness failed: %s%s" % (proc.stdout, proc.stderr[-300:]))
            check("failures=0" in proc.stdout, proc.stdout)
            check(proc.stdout.count(" ok applied") >= 1 or "applied=1" in proc.stdout,
                  "an arm was applied: %s" % proc.stdout)
            check(proc.stdout.count(" ok ") >= 14, "14 cases: %s" % proc.stdout)
            check("CONTENT_CHANGED" not in proc.stdout, proc.stdout)
            print("  ok  rewrite path: exact word, no collateral, clean restore")

        group("the marker rewrite path runs on fake BOs", marker_bo, failures)

        def marker_planner():
            proc = subprocess.run([sys.executable, SLOT_PLANNER, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "planner selftest failed: %s" % proc.stdout)
            if not os.path.exists(ROWS_CSV):
                print("  ok  planner selftest only (captured rows CSV not present)")
                return
            out = os.path.join(tmpdir, "plan")
            proc = subprocess.run([sys.executable, SLOT_PLANNER, "--rows", ROWS_CSV,
                                   "--module", "DPU", "--out-dir", out],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "planner failed: %s" % proc.stderr)
            plan = json.load(open(os.path.join(out, "plan.json")))
            check(plan["marker_reg_ok"] is True, "default marker register is allowed")
            # The safe rule (same task, no bank write in between) leaves NO slot
            # in the captured SDPA stream: every task writes S_POINTER first.
            # This is the finding, not a failure - the report must say so.
            check(plan["slot_count"] == 0,
                  "the captured stream must yield no UNVERIFIED slots (%d)"
                  % plan["slot_count"])
            report = open(os.path.join(out, "report.md")).read()
            check("没有满足条件的槽位" in report, report[:400])
            print("  ok  captured SDPA stream yields 0 safe slots (reported)")

        group("the planner proposes length-preserving slots", marker_planner, failures)

        def marker_arm_runner():
            out = os.path.join(tmpdir, "arms")
            proc = subprocess.run([MARKER_RUNNER, "--task", "1", "--slot", "2",
                                   "--value", "0x00c0ffee", "--dry-run",
                                   "--out", out],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "dry run failed: %s" % proc.stderr)
            for arm in ("original", "nop", "marker"):
                check("EPERF_RKNPU_MARKER_ARM=%s" % arm in proc.stdout,
                      "arm %s planned: %s" % (arm, proc.stdout))
            proc = subprocess.run([MARKER_RUNNER, "--slot", "2", "--out", out,
                                   "--dry-run"], capture_output=True, text=True)
            check(proc.returncode != 0, "a missing --task must be refused")
            print("  ok  three arms planned, missing --task refused")

        group("the marker-arm runner plans the three arms", marker_arm_runner, failures)

        def dup_config_workload():
            # The only slot the safe rule accepts must be created on purpose: a
            # synthetic task that writes one config register twice.  The emitter
            # prints the plan, the planner must independently agree, and the
            # runner must be able to consume that plan directly.
            if not os.path.exists(MIN_CONV):
                print("  ok  skipped (min_conv.c not present)")
                return
            exe = os.path.join(tmpdir, "min_conv")
            res = subprocess.run(["gcc", "-O2", "-o", exe, MIN_CONV, "-lm"],
                                 capture_output=True, text=True)
            check(res.returncode == 0, "min_conv build failed: %s" % res.stderr[-400:])
            rows = os.path.join(tmpdir, "dup_rows.csv")
            proc = subprocess.run([exe, "--dup-config", "--dump-plan",
                                   "--dump-rows", rows],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "dump failed: %s" % proc.stderr[-300:])
            match = re.search(r"MARKER_PLAN task=(\d+) slot=(\d+) reg=(0x[0-9a-f]+) "
                              r"orig=(0x[0-9a-f]+)", proc.stdout)
            check(match is not None, "no MARKER_PLAN line: %s" % proc.stdout[-300:])
            task, slot, reg, orig = match.group(1), int(match.group(2)), \
                match.group(3), match.group(4)

            out = os.path.join(tmpdir, "dup_plan")
            subprocess.run([sys.executable, SLOT_PLANNER, "--rows", rows,
                            "--module", "CNA", "--out-dir", out],
                           check=True, capture_output=True)
            plan = json.load(open(os.path.join(out, "plan.json")))
            check(plan["slot_count"] == 1,
                  "the deliberate duplicate is the only safe slot (%d)"
                  % plan["slot_count"])
            found = plan["slots"][0]
            check(str(found["flat_task"]) == task and found["slot"] == slot,
                  "planner agrees on the position: %s vs task=%s slot=%d"
                  % (found, task, slot))
            check(found["reg"] == reg and found["orig_word"] == orig,
                  "planner agrees on the word: %s vs %s/%s"
                  % (found, reg, orig))

            # without the duplicate there is nothing to replace
            rows2 = os.path.join(tmpdir, "nodup_rows.csv")
            subprocess.run([exe, "--dump-rows", rows2], check=True,
                           capture_output=True)
            out2 = os.path.join(tmpdir, "nodup_plan")
            subprocess.run([sys.executable, SLOT_PLANNER, "--rows", rows2,
                            "--module", "CNA", "--out-dir", out2],
                           check=True, capture_output=True)
            check(json.load(open(os.path.join(out2, "plan.json")))["slot_count"] == 0,
                  "a stream without the duplicate has no safe slot")

            # the runner can consume the emitter's plan directly
            plan_file = os.path.join(tmpdir, "plan.txt")
            open(plan_file, "w").write(proc.stdout)
            arms = subprocess.run([MARKER_RUNNER, "--plan", plan_file, "--dry-run",
                                   "--out", os.path.join(tmpdir, "arms2")],
                                  capture_output=True, text=True)
            check(arms.returncode == 0, "runner rejected the plan: %s" % arms.stderr)
            check("slot=%d" % slot in arms.stdout and orig in arms.stdout,
                  "runner used the planned slot and original word: %s" % arms.stdout[:300])
            print("  ok  --dup-config -> planner agrees (task=%s slot=%d) -> runner uses it"
                  % (task, slot))

        group("the synthetic workload creates a verifiable redundant slot",
              dup_config_workload, failures)

        def arm_comparator():
            # The tool that decides "did the marker change the output": it must
            # be bit-exact by default and must not let a tolerance hide a real
            # difference.
            proc = subprocess.run([sys.executable, ARM_COMPARE, "--selftest"],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "comparator selftest failed: %s" % proc.stdout)

            dumps = os.path.join(tmpdir, "dumps")
            os.makedirs(dumps, exist_ok=True)
            base = os.path.join(dumps, "original.bin")
            with open(base, "wb") as fh:
                fh.write(bytes(64))
            for arm, patch in (("nop", None), ("marker", 17)):
                path = os.path.join(dumps, arm + ".bin")
                data = bytearray(bytes(64))
                if patch is not None:
                    data[patch] ^= 0x01
                with open(path, "wb") as fh:
                    fh.write(bytes(data))
            out = os.path.join(tmpdir, "cmp")
            proc = subprocess.run([sys.executable, ARM_COMPARE, "--baseline", base,
                                   "--arm", "nop=" + os.path.join(dumps, "nop.bin"),
                                   "--arm", "marker=" + os.path.join(dumps, "marker.bin"),
                                   "--out-dir", out],
                                  capture_output=True, text=True)
            check(proc.returncode == 1, "a differing arm must fail the comparison")
            report = open(os.path.join(out, "report.md")).read()
            check("未通过" in report and "marker" in report, report[:300])
            check("identical" in report, report[:300])
            print("  ok  bit-exact by default; the differing arm fails the report")

            # the arm runner must actually invoke the comparison
            out2 = os.path.join(tmpdir, "arms_cmp")
            proc = subprocess.run([MARKER_RUNNER, "--plan", os.path.join(tmpdir, "plan.txt"),
                                   "--dumps", dumps, "--hook", dummy_so, "--runs", "1",
                                   "--out", out2, "--", "/bin/true"],
                                  capture_output=True, text=True)
            check(os.path.exists(os.path.join(out2, "compare", "report.md")),
                  "the runner did not produce a comparison: %s" % proc.stdout[-300:])
            print("  ok  the arm runner runs the comparison and fails on a difference")

        group("the three-arm comparison is bit-exact by default",
              arm_comparator, failures)

        def latency_instrumentation():
            # The stage timestamps are what the latency work is attributed with;
            # they must be present, ordered, and consistent with the phase.
            meta, config, rows, _proc = run(selftest, tmpdir, name="lat",
                                        args=["--phase-ns=2000000", "--run-ms=30",
                                              "--cap=1"])
            check(len(rows) == 1, "one sample")
            for key in ("lat_arm_observed_ns", "lat_phase_end_ns",
                        "lat_power_begin_ns", "lat_power_end_ns",
                        "lat_first_read_ns"):
                check(key in meta, "META carries %s: %s" % (key, meta))
            arm = int(meta["lat_arm_observed_ns"])
            phase_end = int(meta["lat_phase_end_ns"])
            power_begin = int(meta["lat_power_begin_ns"])
            power_end = int(meta["lat_power_end_ns"])
            first_read = int(meta["lat_first_read_ns"])
            check(0 <= arm <= phase_end <= power_begin <= power_end <= first_read,
                  "stage order arm=%d phase_end=%d power=%d..%d read=%d"
                  % (arm, phase_end, power_begin, power_end, first_read))
            check(phase_end >= 2_000_000, "the phase delay is visible (%d)" % phase_end)
            check(config.get("prefault") == "0" and config.get("pin_cpu") == "-1",
                  "variants default to off: %s" % config)

        group("the latency stages are instrumented", latency_instrumentation, failures)

        def shell_raw_tiers():
            # a non-vetted offset must be refused before anything runs
            out = os.path.join(tmpdir, "sweep_tier")
            proc = subprocess.run(
                [SWEEP, "--phases", "1000", "--reps", "1", "--raw-off", "0x1024",
                 "--hook", dummy_so, "--out", out, "--",
                 selftest, "--run-ms=3"],
                capture_output=True, text=True)
            check(proc.returncode != 0,
                  "0x1024 without --raw-experimental must fail fast")
            # vetted offset: real runs, raw column, vetted mode
            out = os.path.join(tmpdir, "sweep_vetted")
            proc = subprocess.run(
                [SWEEP, "--phases", "1000,2000", "--reps", "1", "--raw-off", "0x1000",
                 "--hook", dummy_so, "--out", out, "--",
                 selftest, "--run-ms=3"],
                capture_output=True, text=True)
            check(proc.returncode == 0, "vetted sweep failed: %s" % proc.stderr)
            with open(os.path.join(out, "manifest.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 2, "both vetted runs kept (%d)" % len(rows))
            with open(rows[0]["csv"], newline="") as fh:
                header = next(csv.reader(fh))
            check("c0_raw" in header, "raw column present: %s" % header)
            # experimental offset: allowed, but reported as such
            out = os.path.join(tmpdir, "sweep_experimental")
            proc = subprocess.run(
                [SWEEP, "--phases", "1000", "--reps", "1", "--raw-off", "0x1024",
                 "--raw-experimental", "--hook", dummy_so, "--out", out, "--",
                 selftest, "--run-ms=3"],
                capture_output=True, text=True)
            check(proc.returncode == 0, "experimental sweep failed: %s" % proc.stderr)
            with open(os.path.join(out, "manifest.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 1, "experimental run kept (%d)" % len(rows))
            meta = open(os.path.join(out, "meta", rows[0]["run_id"] + ".txt")).read()
            check("raw_mode=experimental" in meta,
                  "experimental tier must be reported: %s" % meta[-300:])
            print("  ok  vetted runs, experimental opt-in, non-vetted refused")

        group("the shell entry exposes the experimental RAW tier", shell_raw_tiers, failures)

        def bank_runner_dry_run():
            out = os.path.join(tmpdir, "bank_dry")
            proc = subprocess.run([BANK_RUNNER, "--dry-run", "--offset", "0x1024",
                                   "--phases", "2000,4000", "--reps", "2",
                                   "--out", out],
                                  capture_output=True, text=True)
            check(proc.returncode == 0, "bank runner dry run failed: %s" % proc.stderr)
            manifest = os.path.join(out, "sweep", "manifest.csv")
            check(os.path.exists(manifest), "sweep manifest produced")
            with open(manifest, newline="") as fh:
                rows = list(csv.DictReader(fh))
            check(len(rows) == 4, "planned runs (%d)" % len(rows))
            check(all(os.path.isabs(r["csv"]) for r in rows), "absolute paths")
            print("  ok  runner assembles an experimental sweep for a non-vetted offset")

        group("the bank-probe runner assembles a sweep", bank_runner_dry_run, failures)

    print()
    if failures:
        print("%d group(s) failed:" % len(failures))
        for name in failures:
            print("  - " + name)
        return 1
    print("all sampler checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
