"""The dead-man's switch, level C — tested against the condition it MEANS.

Why this file exists
--------------------
On 2026-08-28 a safety switch shut a freshly-started VM down after 79
seconds because it asked "is the heartbeat old?" instead of "is the
machine on and unattended?" (§12).  A switch is only as good as the
question it asks, and the only way to know which question it asks is to
run it against cases whose right answer is known in advance.

So this file GENERATES the level-C watchdog exactly the way GCE does —
by executing `docs/gcp-session-4/auto_off_startup.sh` — and then runs
the generated loop against a stubbed machine with a fake clock:

    | GPU util | vLLM counter | lease    | must shut down?          |
    |----------|--------------|----------|--------------------------|
    | busy     | -            | expired  | NO  — somebody is computing
    | idle     | -            | future   | NO  — somebody claimed it
    | idle     | advancing    | expired  | NO  — vLLM is serving
    | idle     | frozen       | expired  | YES — and not before IDLE_MIN
    | idle     | unreachable  | expired  | YES — and not before IDLE_MIN

Only two things are substituted in the generated script: the absolute
path `/sbin/shutdown` (a PATH stub cannot shadow an absolute path) and
nothing else.  The `if` that makes the decision is the one that ships.
"""
import os
import subprocess
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTUP = os.path.join(REPO, "docs", "gcp-session-4", "auto_off_startup.sh")

IDLE_MIN = 20          # must match the constant in the startup script
SAMPLE_SEC = 20        # ditto — the fake clock advances by this per iteration


def _read_constant(name):
    """Read `NAME=value` out of the startup script, so the test cannot
    silently disagree with the artifact it is testing."""
    with open(STARTUP) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(name + "="):
                return int(stripped[len(name) + 1:].split()[0])
    raise AssertionError("%s not found in %s" % (name, STARTUP))


def test_constants_match_the_artifact():
    assert _read_constant("IDLE_MIN") == IDLE_MIN
    assert _read_constant("SAMPLE_SEC") == SAMPLE_SEC
    # A lease granted at boot must outlast the idle threshold, or the
    # machine would shut down while the operator is still connecting.
    assert _read_constant("INITIAL_LEASE_MIN") > IDLE_MIN


def _build(tmpdir, util, metrics_mode, lease_offset_sec):
    """Generate the watchdog the way GCE would, then stub the machine.

    util             : what `nvidia-smi` reports, as a string
    metrics_mode     : 'advancing' | 'frozen' | 'unreachable'
    lease_offset_sec : lease instant relative to the fake start time
    """
    work = str(tmpdir)
    stub = os.path.join(work, "stub")
    os.makedirs(stub)

    clock = os.path.join(work, "fake_now")
    start = 1000000000
    with open(clock, "w") as handle:
        handle.write(str(start))

    # --- generate the real scripts, with only the paths redirected ----
    with open(STARTUP) as handle:
        source = handle.read()
    redirected = []
    for line in source.splitlines(True):
        for key in ("MARK", "LEASE", "IDLE_SCRIPT", "LEASE_SCRIPT"):
            if line.startswith(key + "="):
                line = "%s=%s\n" % (key, os.path.join(work, key.lower()))
                break
        redirected.append(line)
    local_startup = os.path.join(work, "startup.sh")
    with open(local_startup, "w") as handle:
        handle.write("".join(redirected))
    subprocess.run(["bash", local_startup], check=False, cwd=work)

    idle_script = os.path.join(work, "idle_script")
    assert os.path.exists(idle_script), "the startup script wrote no watchdog"

    fired = os.path.join(work, "SHUTDOWN_CALLED")
    with open(idle_script) as handle:
        body = handle.read()
    assert "/sbin/shutdown -h now" in body
    body = body.replace("/sbin/shutdown -h now",
                        "%s/shutdown_stub" % stub)
    with open(idle_script, "w") as handle:
        handle.write(body)

    # --- the stubbed machine ------------------------------------------
    def write(name, text):
        path = os.path.join(stub, name)
        with open(path, "w") as handle:
            handle.write(text)
        os.chmod(path, 0o755)

    write("date", (
        '#!/bin/bash\n'
        'now=$(cat %s)\n'
        'if [ "$1" = "+%%s" ]; then echo "$now"; else echo "FAKE-$now"; fi\n'
    ) % clock)
    write("sleep", (
        '#!/bin/bash\n'
        'now=$(cat %s)\n'
        'echo $(( now + $1 )) > %s\n'
    ) % (clock, clock))
    write("nvidia-smi", '#!/bin/bash\necho "%s"\n' % util)
    if metrics_mode == "advancing":
        curl = ('#!/bin/bash\n'
                'echo "vllm:request_success_total{m=\\"x\\"} $(cat %s)"\n' % clock)
    elif metrics_mode == "frozen":
        curl = ('#!/bin/bash\n'
                'echo "vllm:request_success_total{m=\\"x\\"} 7904"\n')
    else:
        curl = '#!/bin/bash\nexit 7\n'
    write("curl", curl)
    write("shutdown_stub",
          '#!/bin/bash\ncat %s > %s\n' % (clock, fired))

    with open(os.path.join(work, "lease"), "w") as handle:
        handle.write(str(start + lease_offset_sec))

    env = dict(os.environ)
    env["PATH"] = stub + os.pathsep + env["PATH"]
    return idle_script, env, clock, fired, start


def _run(idle_script, env, seconds=6):
    try:
        subprocess.run(["bash", idle_script], env=env,
                       timeout=seconds, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return "exited"
    except subprocess.TimeoutExpired:
        return "still running"


@pytest.mark.parametrize("util,metrics,lease_offset,label", [
    ("87", "unreachable", -1, "the GPU is computing"),
    ("0", "unreachable", 10 ** 6, "somebody holds a lease"),
    ("0", "advancing", -1, "vLLM keeps completing requests"),
])
def test_does_not_shut_down_while_something_says_it_is_in_use(
        tmp_path, util, metrics, lease_offset, label):
    idle_script, env, _clock, fired, _start = _build(
        tmp_path, util, metrics, lease_offset)
    _run(idle_script, env)
    assert not os.path.exists(fired), (
        "level C shut the machine down while %s" % label)


@pytest.mark.parametrize("metrics,label", [
    ("frozen", "the server is up but nobody is sweeping"),
    ("unreachable", "no server at all"),
])
def test_shuts_down_when_unattended_and_not_before_the_threshold(
        tmp_path, metrics, label):
    idle_script, env, _clock, fired, start = _build(
        tmp_path, "0", metrics, -1)
    outcome = _run(idle_script, env, seconds=20)
    assert os.path.exists(fired), (
        "level C never fired although %s and the lease had expired" % label)
    assert outcome == "exited", "the watchdog kept running after shutting down"
    with open(fired) as handle:
        when = int(handle.read().strip())
    waited = when - start
    assert waited >= IDLE_MIN * 60, (
        "fired after only %d s — the 2026-08-28 failure mode, a switch that "
        "shuts down a machine somebody is about to use" % waited)
    assert waited < (IDLE_MIN * 60) + (3 * SAMPLE_SEC), (
        "fired %d s late; the threshold is not being honoured" % waited)


def test_the_lease_helper_writes_an_instant_in_the_future(tmp_path):
    idle_script, env, _clock, _fired, _start = _build(
        tmp_path, "0", "unreachable", -1)
    helper = os.path.join(str(tmp_path), "lease_script")
    assert os.path.exists(helper)
    lease_file = os.path.join(str(tmp_path), "lease")
    before = time.time()
    subprocess.run(["bash", helper, "45"], check=True,
                   stdout=subprocess.DEVNULL)
    with open(lease_file) as handle:
        until = int(handle.read().strip())
    assert until > before + 44 * 60
    assert until < before + 46 * 60


def test_generation_expands_nothing_it_did_not_mean_to(tmp_path):
    """The watchdog is written by a heredoc inside another script.

    An unquoted heredoc runs backticks and `$(...)` AT GENERATION TIME.
    On 2026-08-30 a comment reading ``nvidia-smi`` did exactly that: the
    startup script tried to execute nvidia-smi and curl while writing the
    file, and the comment reached disk with the two names deleted.  It
    was harmless there and would not be somewhere else, so it is pinned:
    generating the switch must produce no stderr and no missing text.
    """
    work = str(tmp_path)
    with open(STARTUP) as handle:
        source = handle.read()
    redirected = []
    for line in source.splitlines(True):
        for key in ("MARK", "LEASE", "IDLE_SCRIPT", "LEASE_SCRIPT"):
            if line.startswith(key + "="):
                line = "%s=%s\n" % (key, os.path.join(work, key.lower()))
                break
        redirected.append(line)
    local_startup = os.path.join(work, "startup.sh")
    with open(local_startup, "w") as handle:
        handle.write("".join(redirected))

    result = subprocess.run(["bash", local_startup], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.stderr == b"", (
        "generating the switch wrote to stderr: %r" % result.stderr[:400])

    with open(os.path.join(work, "idle_script")) as handle:
        generated = handle.read()
    # the two tool names must survive into the generated file
    assert "nvidia-smi --query-gpu=utilization.gpu" in generated
    assert "http://127.0.0.1:8000/metrics" in generated
    assert "$(command -v nvidia-smi" in generated
