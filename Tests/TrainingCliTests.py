"""End-to-end checks for real checkpoints, scheduling, and viewer parity."""
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

executable = Path(sys.argv[1]).resolve()
viewer = sys.argv[2] == "1"


def run(*args, success=True, cwd=None):
    result = subprocess.run([str(executable), *map(str, args)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "SDL_VIDEODRIVER": "dummy"}, cwd=cwd, timeout=30)
    assert (result.returncode == 0) == success, result.stdout
    return result.stdout


with tempfile.TemporaryDirectory(prefix="invaders-test-") as directory:
    root = Path(directory)
    common = ("train", "--generations", 4, "--population", 16, "--elite", 2,
              "--seed", 42, "--max-frames", 120, "--episodes", 1)
    serial = run(*common, "--threads", 1, "--save", root / "serial")
    parallel = run(*common, "--threads", 4, "--save", root / "parallel")
    assert (root / "serial").read_bytes() == (root / "parallel").read_bytes(), "Scheduling changed champion"
    score_pattern = r"Generation \d+: best=[\d.]+ mean=[\d.]+"
    assert re.findall(score_pattern, serial) == re.findall(score_pattern, parallel), "Scheduling changed fitness"
    assert not (root / "serial.tmp").exists(), "Checkpoint replacement left a temporary file"

    # Uneven populations and multiple seeds exercise dynamic scheduling, including
    # more requested workers than agents. Compare every saved improvement too.
    scheduling = ("train", "--generations", 9, "--population", 17, "--elite", 2,
                  "--seed", 42, "--max-frames", 1000, "--episodes", 3)
    reference = run(*scheduling, "--threads", 1, "--save", root / "reference")
    for workers in (3, 8, 32):
        checkpoint = root / f"workers-{workers}"
        output = run(*scheduling, "--threads", workers, "--save", checkpoint)
        assert f"{min(workers, 17)} workers" in output
        assert checkpoint.read_bytes() == (root / "reference").read_bytes(), "Worker count changed champion"
        assert re.findall(score_pattern, output) == re.findall(score_pattern, reference), "Worker count changed fitness"
        assert re.findall(r"mean score ([\d.]+)", output) == re.findall(r"mean score ([\d.]+)", reference), "Scheduling changed improvements"

    # Very short jobs stress generation barriers and idle workers in the pool.
    short = run("train", "--generations", 205, "--population", 7, "--threads", 7,
                "--max-frames", 1, "--episodes", 1, "--save", root / "short")
    assert re.findall(r"Generation (\d+):", short) == [str(n) for n in range(10, 201, 10)] + ["205"]
    # A save failure must shut down all waiting workers and return a normal error.
    failed_save = run(*common, "--threads", 4, "--fresh",
                      "--save", root / "missing-directory" / "best", success=False)
    assert "Error:" in failed_save, failed_save

    explicit = run(*common, "--threads", 2, "--load", root / "serial", "--save", root / "resumed")
    shutil.copyfile(root / "serial", root / "automatic")
    automatic = run(*common, "--threads", 2, "--save", root / "automatic")
    assert "Starting from checkpoint:" in automatic
    assert re.findall(score_pattern, automatic) == re.findall(score_pattern, explicit), "Automatic resume did not seed the saved champion"
    assert (root / "automatic").read_bytes() == (root / "resumed").read_bytes()
    default_run = root / "default"
    default_run.mkdir()
    shutil.copyfile(root / "serial", default_run / "invaders_best")
    default = run(*common, "--threads", 2, cwd=default_run)
    assert "Starting from checkpoint: invaders_best" in default
    assert re.findall(score_pattern, default) == re.findall(score_pattern, explicit)

    # With one frame every agent scores zero. An equal score must leave the
    # previously saved champion intact, including its modification time.
    tied = root / "tied"
    tied.write_text("1\n1\n1\n0\n1\n")
    os.utime(tied, ns=(1_000_000_000, 1_000_000_000))
    before = (tied.read_bytes(), tied.stat().st_mtime_ns)
    unchanged = run("train", "--generations", 3, "--population", 8, "--max-frames", 1,
                    "--episodes", 1, "--save", tied)
    assert (tied.read_bytes(), tied.stat().st_mtime_ns) == before, "A tie replaced the saved best"
    assert "Saved new best:" not in unchanged
    copied_tie = root / "copied-tie"
    run("train", "--generations", 3, "--population", 8, "--max-frames", 1,
        "--episodes", 1, "--load", tied, "--save", copied_tie)
    assert copied_tie.read_bytes() == before[0], "Random parent ties changed the explicitly loaded checkpoint on equal fitness"
    fresh = run(*common, "--threads", 1, "--fresh", "--save", root / "automatic")
    assert "Starting a fresh population" in fresh
    assert (root / "automatic").read_bytes() == (root / "serial").read_bytes(), "--fresh still loaded the checkpoint"
    run("train", "--fresh", "--load", root / "serial", success=False)
    run("watch", "--fresh", success=False)

    # The checkpoint must become usable while training is still running.
    ongoing = root / "ongoing"
    with (root / "ongoing.log").open("w") as log:
        process = subprocess.Popen([str(executable), "train", "--generations", "10000000",
                                    "--population", "8", "--max-frames", "120", "--threads", "1",
                                    "--save", str(ongoing)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5
            while not ongoing.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ongoing.exists() and process.poll() is None, "Best was not saved during training"
        finally:
            process.terminate()
            process.wait(timeout=5)
    run(*common, "--load", ongoing, "--save", root / "after-interruption")

    # Indefinite mode must pass the normal 100-generation limit and stop cleanly.
    forever = root / "forever"
    with (root / "forever.log").open("w+") as log:
        process = subprocess.Popen([str(executable), "train", "--indefinite", "--population", "8",
                                    "--max-frames", "1", "--episodes", "1", "--threads", "2",
                                    "--save", str(forever)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            output = ""
            while process.poll() is None and time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                if "Generation 110:" in output: break
                time.sleep(0.01)
            assert "Generation 110:" in output and process.poll() is None, output
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) == 0
            log.seek(0)
            output = log.read()
            assert "Stopped after Ctrl+C." in output and "Best checkpoint:" in output, output
            assert forever.exists() and not (root / "forever.tmp").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    run(*common, "--load", forever, "--save", root / "forever-reloaded")

    # Interrupt the first generation before any agent finishes all episodes.
    # A pre-existing champion must remain usable and must not be overwritten.
    interrupted = root / "interrupted"
    shutil.copyfile(root / "serial", interrupted)
    before_interruption = interrupted.read_bytes()
    for checkpoint, expected_message in ((interrupted, "Existing checkpoint retained:"),
                                         (root / "interrupted-fresh", "No agent completed evaluation; no new checkpoint saved.")):
        with (root / (checkpoint.name + ".log")).open("w+") as log:
            process = subprocess.Popen([str(executable), "train", "--indefinite", "--population", "4",
                                        "--max-frames", "3600", "--episodes", "10000", "--threads", "2",
                                        "--save", str(checkpoint)], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                output = ""
                while process.poll() is None and time.monotonic() < deadline:
                    log.seek(0)
                    output = log.read()
                    if "Training indefinitely;" in output: break
                    time.sleep(0.01)
                assert "Training indefinitely;" in output, output
                process.send_signal(signal.SIGINT)
                assert process.wait(timeout=5) == 0
                log.seek(0)
                output = log.read()
                assert "Stopped after Ctrl+C." in output and expected_message in output, output
                if checkpoint == interrupted:
                    assert checkpoint.read_bytes() == before_interruption
                else:
                    assert not checkpoint.exists(), "A partial evaluation was saved as a champion"
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
    run("train", "--indefinite", "--generations", 2, success=False)
    run("watch", "--indefinite", success=False)
    run("train", "--load", root / "missing", success=False)
    run("train", "--threads", 0, success=False)
    run("watch", "--fps", 0, success=False)
    run("train", "--population", 2, "--elite", 2, success=False)
    invalid = root / "invalid"
    for content in ("", "0\n", "1\n1\n99\n0\n0\n", "1\n1\n1\n-1\n0\n", "1\n1\n1\n0\n0\nextra"):
        invalid.write_text(content)
        run("train", "--load", invalid, success=False)
        run("train", "--save", invalid, success=False)
        assert invalid.read_text() == content, "Invalid automatic checkpoint was overwritten"
    if viewer:
        playback = run("watch", "--load", root / "serial", "--seed", 42,
                       "--max-frames", 120, "--fps", 1000)
        best = float(re.findall(r"mean score ([\d.]+)", serial)[-1])
        watched = int(re.search(r"score=(\d+)", playback).group(1))
        assert best == watched, f"Viewer score {watched} differs from training fitness {best}"
        run("play", "--max-frames", 2, "--fps", 1000)
    else:
        run("watch", "--max-frames", 2, success=False)
print("PASS: parallel determinism, automatic resume, ongoing saves, checkpoint validation, CLI errors, and playback")
