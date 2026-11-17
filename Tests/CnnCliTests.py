"""CNN training/checkpoints and playback through the real command-line application."""
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


with tempfile.TemporaryDirectory(prefix="invaders-cnn-test-") as directory:
    root = Path(directory)
    common = ("train", "--generations", 3, "--population", 8, "--elite", 2,
              "--seed", 42, "--max-frames", 120, "--episodes", 2)
    serial = run(*common, "--agent", "cnn", "--threads", 1, "--save", root / "serial")
    assert "13598 parameters" in serial
    assert "softmax sampling (temperature=1)" in serial
    score_pattern = r"Generation \d+: best=[\d.]+ mean=[\d.]+"
    saved = (root / "serial").read_bytes()
    assert saved.startswith(b"EVOLUTION_CNN 1\n")
    for threads in (3, 16):
        checkpoint = root / f"parallel-{threads}"
        output = run(*common, "--agent", "cnn", "--threads", threads, "--save", checkpoint)
        assert checkpoint.read_bytes() == saved, "Scheduling changed CNN weights"
        assert re.findall(score_pattern, serial) == re.findall(score_pattern, output), "Scheduling changed CNN fitness"
    assert not (root / "serial.tmp").exists()

    # Long enough for different policies to end at different times. Odd-sized
    # populations exercise compacted batches, slot refilling, and multiple seeds.
    long_common = ("train", "--agent", "cnn", "--generations", 3, "--population", 11,
                   "--elite", 2, "--seed", 123, "--max-frames", 1800, "--episodes", 3)
    unbatched = run(*long_common, "--threads", 1, "--batch-size", 1, "--save", root / "unbatched")
    for threads, batch in ((1, 4), (3, 4), (4, 16), (2, 1)):
        checkpoint = root / f"batch-{threads}-{batch}"
        output = run(*long_common, "--threads", threads, "--batch-size", batch, "--save", checkpoint)
        assert checkpoint.read_bytes() == (root / "unbatched").read_bytes(), "Batching changed the saved champion"
        assert re.findall(score_pattern, unbatched) == re.findall(score_pattern, output), "Batching changed population fitness"
    for batch in (0, -1, 257, "bad"):
        run("train", "--agent", "cnn", "--batch-size", batch, success=False)
    run("train", "--agent", "program", "--batch-size", 4, success=False)
    run("watch", "--agent", "cnn", "--load", root / "serial", "--batch-size", 4, success=False)

    # Type detection must work for both explicit loading and automatic resuming.
    explicit = run(*common, "--load", root / "serial", "--save", root / "explicit")
    shutil.copyfile(root / "serial", root / "automatic")
    automatic = run(*common, "--save", root / "automatic")
    assert "CPU CNN:" in automatic and "Starting from checkpoint:" in automatic
    assert re.findall(score_pattern, explicit) == re.findall(score_pattern, automatic)
    assert (root / "explicit").read_bytes() == (root / "automatic").read_bytes()

    # No score is possible in one frame. This checks exact float round trips and
    # that neutral parent changes never replace an equally scoring saved champion.
    roundtrip = root / "roundtrip"
    run("train", "--load", root / "serial", "--save", roundtrip, "--generations", 1,
        "--population", 4, "--max-frames", 1, "--episodes", 1)
    assert roundtrip.read_bytes() == saved, "CNN float serialization lost precision"
    os.utime(roundtrip, ns=(1_000_000_000, 1_000_000_000))
    tied = run("train", "--save", roundtrip, "--generations", 3,
               "--population", 4, "--max-frames", 1, "--episodes", 1)
    assert roundtrip.read_bytes() == saved and roundtrip.stat().st_mtime_ns == 1_000_000_000
    assert "Saved new best:" not in tied

    default = root / "default"
    default.mkdir()
    program = default / "invaders_best"
    program.write_text("1\n1\n1\n0\n1\n")
    run(*common, "--agent", "cnn", cwd=default)
    assert (default / "invaders_cnn_best").read_bytes() == saved
    assert program.read_text() == "1\n1\n1\n0\n1\n", "CNN training overwrote the program checkpoint"
    assert "Starting from checkpoint:" in run(*common, "--agent", "cnn", cwd=default)
    run(*common, "--agent", "cnn", "--fresh", "--save", root / "automatic")
    assert (root / "automatic").read_bytes() == saved

    for name, value in (("--agent", "bad"), ("--mutation-rate", "0"), ("--mutation-rate", "1.1"),
                        ("--mutation-rate", "nan"), ("--mutation-sigma", "0"),
                        ("--mutation-sigma", "inf"), ("--mutation-sigma", "9")):
        run("train", "--agent", "cnn", name, value, success=False)
    run("train", "--agent", "program", "--mutation-rate", 0.1, success=False)
    run("train", "--agent", "program", "--load", root / "serial", success=False)
    run("train", "--agent", "cnn", "--save", program, success=False)
    run(*common, "--agent", "cnn", "--save", root / "missing" / "best", success=False)

    valid = saved.decode().splitlines()
    invalid = root / "invalid"
    broken = ["", "EVOLUTION_CNN 2\n", "EVOLUTION_CNN 1\n1 1 1\n",
              "\n".join(valid[:-1]), saved.decode() + "extra\n"]
    for bad_weight in ("nan", "inf", "9", "-9"):
        lines = valid.copy()
        lines[2] = bad_weight
        broken.append("\n".join(lines))
    for content in broken:
        invalid.write_text(content)
        run("train", "--agent", "cnn", "--save", invalid, success=False)
        assert invalid.read_text() == content, "Malformed CNN checkpoint was overwritten"

    assert "CPU CNN" in run("benchmark", "--load", root / "serial", "--frames", 10)

    # Stop a partially evaluated CNN generation and preserve its loaded champion.
    interrupted = root / "interrupted-cnn"
    shutil.copyfile(root / "serial", interrupted)
    with (root / "interrupted-cnn.log").open("w+") as log:
        process = subprocess.Popen([str(executable), "train", "--indefinite", "--population", "4",
                                    "--max-frames", "3600", "--episodes", "10000", "--threads", "2",
                                    "--save", str(interrupted)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            output = ""
            while process.poll() is None and time.monotonic() < deadline:
                log.seek(0)
                output = log.read()
                if "Training indefinitely;" in output: break
                time.sleep(0.01)
            assert "CPU CNN:" in output and "Training indefinitely;" in output, output
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=5) == 0
            log.seek(0)
            output = log.read()
            assert "Stopped after Ctrl+C." in output and "Existing checkpoint retained:" in output, output
            assert interrupted.read_bytes() == saved, "Ctrl+C replaced an unevaluated CNN checkpoint"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    if viewer:
        output = run("watch", "--load", root / "serial", "--seed", 42,
                     "--max-frames", 120, "--episodes", 2, "--fps", 1000)
        scores = list(map(int, re.findall(r"score=(\d+)", output)))
        expected = float(re.findall(r"mean score ([\d.]+)", serial)[-1])
        assert len(scores) == 2 and sum(scores) / 2 == expected, "CNN viewer and training disagree"
        run("watch", "--agent", "cnn", "--max-frames", 1, cwd=default)
    else:
        run("watch", "--load", root / "serial", "--max-frames", 1, success=False)
print("PASS: CNN parallel training, resume, checkpoints, validation, CLI, and playback")
