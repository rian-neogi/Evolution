"""ES training through the real CLI: updates, deterministic evaluation, and saves."""
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

executable = Path(sys.argv[1]).resolve()
viewer = sys.argv[2] == "1"


def run(*args, cwd=None, success=True):
    result = subprocess.run([str(executable), *map(str, args)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd,
                            env={**os.environ, "SDL_VIDEODRIVER": "dummy"}, timeout=30)
    assert (result.returncode == 0) == success, result.stdout
    return result.stdout


def stop_after(args, path, marker):
    with path.open("w+") as log:
        process = subprocess.Popen([str(executable), *map(str, args)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            output = ""
            while process.poll() is None and time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                if marker in output:
                    break
                time.sleep(.01)
            assert marker in output and process.poll() is None, output
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) == 0
            log.seek(0)
            output = log.read()
            assert "Stopped after Ctrl+C." in output, output
            return output
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


with tempfile.TemporaryDirectory(prefix="invaders-cnn-es-test-") as directory:
    root = Path(directory)
    common = ("train", "--agent", "cnn", "--optimizer", "es", "--generations", 4,
              "--population", 8, "--seed", 42, "--max-frames", 600, "--episodes", 2,
              "--weight-decay", 0)
    serial = run(*common, "--threads", 1, "--batch-size", 1, "--save", root / "serial")
    saved = (root / "serial").read_bytes()
    assert saved.startswith(b"EVOLUTION_CNN 1\n")
    assert "centered ranks + Adam" in serial and "4 mirrored pairs" in serial, serial
    assert "Generation 4:" in serial, serial
    assert any(float(g) > 0 for g in re.findall(r"gradient-norm=([\d.e+-]+)", serial)), serial
    assert any(float(s) > 0 for s in re.findall(r"step-norm=([\d.e+-]+)", serial)), serial
    metric_pattern = r"Generation \d+: center=.*? step-norm=[\d.e+-]+"
    for threads, batch in ((1, 4), (3, 4), (16, 256)):
        checkpoint = root / f"parallel-{threads}-{batch}"
        output = run(*common, "--threads", threads, "--batch-size", batch, "--save", checkpoint)
        assert checkpoint.read_bytes() == saved, "Worker/batch count changed ES weights"
        assert re.findall(metric_pattern, output) == re.findall(metric_pattern, serial), "Worker/batch count changed ES updates"
    assert not (root / "serial.tmp").exists()

    # An existing GA checkpoint is also a valid ES center; exercise inferred type.
    initial = root / "initial"
    run("train", "--agent", "cnn", "--seed", 42, "--fresh", "--generations", 1,
        "--population", 2, "--max-frames", 1, "--episodes", 1, "--save", initial)
    initial_bytes = initial.read_bytes()
    resume = ("train", "--optimizer", "es", "--generations", 3, "--population", 8,
              "--seed", 42, "--max-frames", 300, "--episodes", 2)
    explicit = run(*resume, "--load", initial, "--save", root / "explicit")
    shutil.copyfile(initial, root / "automatic")
    automatic = run(*resume, "--save", root / "automatic")
    assert "Starting from checkpoint:" in automatic and "Adam state starts fresh" in automatic
    assert re.findall(metric_pattern, explicit) == re.findall(metric_pattern, automatic)
    assert (root / "explicit").read_bytes() == (root / "automatic").read_bytes()

    # All scores tie. The center still changes through L2/Adam, but only an
    # improved, fully evaluated center may replace the initial best checkpoint.
    tied = root / "tied"
    one_frame = ("train", "--optimizer", "es", "--generations", 3,
                 "--population", 4, "--max-frames", 1, "--episodes", 1)
    output = run(*one_frame, "--load", initial, "--save", tied)
    assert tied.read_bytes() == initial_bytes, "A tied updated center replaced the initial best"
    assert all(float(g) == 0 for g in re.findall(r"gradient-norm=([\d.e+-]+)", output)), output
    assert all(float(s) > 0 for s in re.findall(r"step-norm=([\d.e+-]+)", output)), output
    os.utime(tied, ns=(1_000_000_000, 1_000_000_000))
    for extra in ((), ("--load", tied)):
        output = run(*one_frame, *extra, "--save", tied)
        assert tied.read_bytes() == initial_bytes and tied.stat().st_mtime_ns == 1_000_000_000
        assert "Saved new best:" not in output

    # Explicit fresh starts ignore a pre-existing policy and use the default CNN path.
    default = root / "default"
    default.mkdir()
    run(*common, "--fresh", cwd=default)
    assert (default / "invaders_cnn_best").read_bytes() == saved
    shutil.copyfile(initial, root / "fresh")
    run(*common, "--fresh", "--save", root / "fresh")
    assert (root / "fresh").read_bytes() == saved

    for flag, values in (("--es-sigma", (0, -1, 9, "nan", "inf", "bad", "0.1junk")),
                         ("--learning-rate", (0, -1, 2, "nan", "inf")),
                         ("--weight-decay", (-1, 2, "nan", "inf"))):
        for value in values:
            run(*common, flag, value, success=False)
    for extra in (("--population", 3), ("--elite", 1), ("--mutation-rate", .1),
                  ("--mutation-sigma", .05), ("--optimizer", "bad"), ("--agent", "program")):
        run(*common, *extra, success=False)
    for flag in ("--es-sigma", "--learning-rate", "--weight-decay"):
        run("train", "--agent", "cnn", flag, .01, success=False)
    run("watch", "--load", initial, "--optimizer", "es", success=False)
    run("benchmark", "--load", initial, "--learning-rate", .01, success=False)
    run(*common, "--save", root / "missing-dir" / "best", success=False)
    invalid = root / "invalid"
    invalid.write_text("EVOLUTION_CNN 1\ninvalid\n")
    run(*common, "--save", invalid, success=False)
    assert invalid.read_text() == "EVOLUTION_CNN 1\ninvalid\n"

    # Interrupt an unfinished baseline, both with and without an existing policy.
    for existing in (False, True):
        checkpoint = root / ("interrupted-existing" if existing else "interrupted-fresh")
        if existing:
            shutil.copyfile(initial, checkpoint)
        output = stop_after(("train", "--agent", "cnn", "--optimizer", "es", "--indefinite",
                             "--population", 4, "--threads", 2, "--max-frames", 3600,
                             "--episodes", 10000, "--save", checkpoint),
                            root / f"interrupted-{existing}.log", "Training indefinitely;")
        if existing:
            assert "Existing checkpoint retained:" in output and checkpoint.read_bytes() == initial_bytes
        else:
            assert "No agent completed evaluation" in output and not checkpoint.exists()

    # Interrupt during perturbation evaluation, after a completed center was
    # saved. Neither a partial population nor an unmatched pair may drive Adam.
    partial = root / "partial-population"
    output = stop_after(("train", "--optimizer", "es", "--indefinite", "--load", initial,
                         "--population", 128, "--threads", 2, "--max-frames", 3600,
                         "--episodes", 10, "--save", partial), root / "partial.log",
                        "Evaluating ES generation 1:")
    assert "update skipped" in output and "Generation 1: center=" not in output, output
    assert partial.read_bytes() == initial_bytes and not (root / "partial-population.tmp").exists()

    # A completed baseline is saved before perturbation training, and indefinite
    # mode advances beyond the default generation limit and exits cleanly.
    forever = root / "forever"
    output = stop_after(("train", "--agent", "cnn", "--optimizer", "es", "--indefinite",
                         "--population", 2, "--threads", 2, "--max-frames", 1,
                         "--episodes", 1, "--save", forever), root / "forever.log", "Generation 110:")
    assert "Best checkpoint:" in output and forever.exists()
    assert not (root / "forever.tmp").exists()
    run("benchmark", "--load", forever, "--frames", 10)

    if viewer:
        output = run("watch", "--load", root / "serial", "--seed", 42,
                     "--max-frames", 600, "--episodes", 2, "--fps", 1000)
        scores = list(map(int, re.findall(r"score=(\d+)", output)))
        expected = float(re.findall(r"mean score ([\d.]+)", serial)[-1])
        assert len(scores) == 2 and sum(scores) / 2 == expected, "Saved ES center differs from its evaluated score"

print("PASS: ES updates, batching/thread determinism, GA checkpoint loading, ties, interruption, and CLI")
