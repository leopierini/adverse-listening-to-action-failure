"""The persistent start-retry tick, tested against a stubbed gcloud.

Why this file exists
--------------------
A retry loop that outlives the session can also outlive its purpose: if
it kept sending `start` after the machine had been on and shut itself
down, it would rent an A100 every ten minutes with nobody watching.  So
the tick's stopping conditions are not a comment, they are tests.

The stub records every gcloud invocation, which is how "sent no start"
is asserted rather than assumed.
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK = os.path.join(REPO, "scripts", "vm_start_retry.sh")


def _stub_gcloud(tmp_path, status, calls):
    """A gcloud that answers `describe` with `status` and logs its argv."""
    path = os.path.join(str(tmp_path), "gcloud")
    with open(path, "w") as handle:
        handle.write(
            '#!/bin/bash\n'
            'echo "$@" >> %s\n'
            'case "$*" in\n'
            '  *describe*lastStartTimestamp*) echo "2026-08-28T15:02:27	2026-08-28T17:33:21" ;;\n'
            '  *describe*) echo "%s" ;;\n'
            '  *start*) echo "operation-1756000000000-abc" ;;\n'
            'esac\n' % (calls, status))
    os.chmod(path, 0o755)
    return path


def _run(tmp_path, status, deadline_offset=3600, stop=None, attempts=None):
    state = os.path.join(str(tmp_path), "state")
    os.makedirs(state)
    calls = os.path.join(str(tmp_path), "calls")
    gcloud = _stub_gcloud(tmp_path, status, calls)

    import time
    with open(os.path.join(state, "deadline"), "w") as handle:
        handle.write(str(int(time.time()) + deadline_offset))
    if stop is not None:
        with open(os.path.join(state, "stop"), "w") as handle:
            handle.write(stop)
    if attempts is not None:
        with open(os.path.join(state, "attempts"), "w") as handle:
            handle.write(str(attempts))

    env = dict(os.environ)
    env["THESIS_GCLOUD_BIN"] = gcloud
    env["THESIS_VM_RETRY_STATE"] = state
    result = subprocess.run(["bash", TICK], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr
    argv = ""
    if os.path.exists(calls):
        with open(calls) as handle:
            argv = handle.read()
    log = ""
    log_path = os.path.join(state, "retry.log")
    if os.path.exists(log_path):
        with open(log_path) as handle:
            log = handle.read()
    return state, argv, log


def test_terminated_sends_one_async_start(tmp_path):
    state, argv, log = _run(tmp_path, "TERMINATED")
    assert argv.count("instances start") == 1
    assert "--async" in argv, "REGOLA 2: lo start deve essere --async"
    assert not os.path.exists(os.path.join(state, "stop"))
    with open(os.path.join(state, "attempts")) as handle:
        assert handle.read().strip() == "1"
    assert "tentativo 1" in log


def test_running_stops_the_loop_and_sends_no_start(tmp_path):
    state, argv, log = _run(tmp_path, "RUNNING")
    assert "instances start" not in argv, (
        "il tick ha inviato uno start su una macchina gia' accesa")
    with open(os.path.join(state, "stop")) as handle:
        assert handle.read().strip() == "running"
    assert "RUNNING" in log


@pytest.mark.parametrize("status", ["STAGING", "PROVISIONING", "STOPPING",
                                    "SUSPENDING", "REPAIRING"])
def test_transitional_states_do_not_stack_a_second_start(tmp_path, status):
    state, argv, log = _run(tmp_path, status)
    assert "instances start" not in argv, (
        "REGOLA 3: start inviato sopra uno stato %s" % status)
    assert not os.path.exists(os.path.join(state, "stop")), (
        "uno stato transitorio non deve fermare il ritentativo")
    assert "transitorio" in log


def test_a_written_stop_file_makes_the_tick_a_no_op(tmp_path):
    state, argv, log = _run(tmp_path, "TERMINATED", stop="running")
    assert argv == "", (
        "con `stop` presente il tick non deve nemmeno interrogare gcloud")
    assert log == ""


def test_an_expired_deadline_stops_the_loop(tmp_path):
    state, argv, log = _run(tmp_path, "TERMINATED", deadline_offset=-1)
    assert argv == "", "scaduto: non deve chiamare gcloud"
    with open(os.path.join(state, "stop")) as handle:
        assert handle.read().strip() == "scadenza"
    assert "scadenza raggiunta" in log


def test_an_unreadable_status_retries_instead_of_guessing(tmp_path):
    """REGOLA 1 in its strongest form: no answer is not `TERMINATED`."""
    state, argv, log = _run(tmp_path, "")
    assert "instances start" not in argv, (
        "ha inviato uno start senza sapere in che stato fosse la macchina")
    assert not os.path.exists(os.path.join(state, "stop"))
    assert "ILLEGGIBILE" in log


def test_attempts_accumulate_across_ticks(tmp_path):
    state, _argv, log = _run(tmp_path, "TERMINATED", attempts=22)
    with open(os.path.join(state, "attempts")) as handle:
        assert handle.read().strip() == "23"
    assert "tentativo 23" in log


def test_the_deadline_outlives_a_self_shutdown_window():
    """The tick must be able to SEE the machine it started.

    Level C keeps an unattended VM alive at least INITIAL_LEASE_MIN
    minutes.  If the retry interval were longer than that, the machine
    could boot and shut itself down entirely between two ticks, the loop
    would never write `stop`, and it would start it again forever.
    """
    switch = os.path.join(REPO, "docs", "gcp-session-4",
                          "auto_off_startup.sh")
    lease_min = None
    with open(switch) as handle:
        for line in handle:
            if line.strip().startswith("INITIAL_LEASE_MIN="):
                lease_min = int(line.strip().split("=")[1].split()[0])
    assert lease_min is not None
    ctl = os.path.join(REPO, "scripts", "vm_retry_ctl.sh")
    interval = None
    with open(ctl) as handle:
        for line in handle:
            if line.strip().startswith("INTERVAL="):
                interval = int(line.split(":-")[1].split("}")[0])
    assert interval is not None
    assert interval < lease_min * 60, (
        "intervallo %ds >= lease iniziale %dmin: la macchina potrebbe "
        "accendersi e spegnersi fra due tick" % (interval, lease_min * 60))
