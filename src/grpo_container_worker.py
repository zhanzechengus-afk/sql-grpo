import json
import os
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import tempfile


def normalize(value):
    if isinstance(value, (tuple, list)):
        return "\n".join(normalize(item) for item in value)
    return "\n".join(line.rstrip() for line in str(value).strip().splitlines())


def limit_child():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256_000, 256_000))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def verify_one(candidate, setup, test):
    source = setup + "\n" + candidate + "\n"
    marker = secrets.token_hex(24)
    kind = test["kind"]
    if kind == "assert":
        source += test["code"] + "\n"
    elif kind == "function":
        name = test["function_name"]
        if not name.isidentifier():
            raise ValueError("invalid function name")
        value = test["input"]
        call = "**" + repr(value) if isinstance(value, dict) else "*" + repr(value) if isinstance(value, list) else repr(value)
        source += (
            f"_evaluate = globals().get({name!r})\n"
            f"if _evaluate is None: _evaluate = getattr(Solution(), {name!r})\n"
            f"assert _evaluate({call}) == {test['expected']!r}\n"
        )
    elif kind != "stdin":
        raise ValueError("unknown test kind")
    if kind != "stdin":
        source += f"print({marker!r})\n"
    stdin = normalize(test.get("input", "")) if kind == "stdin" else ""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "candidate.py").write_text(source)
        with tempfile.TemporaryFile() as output:
            try:
                run = subprocess.run([sys.executable, "-I", "-B", "candidate.py"], input=stdin,
                                     text=True, stdout=output, stderr=subprocess.DEVNULL,
                                     cwd=tmp, preexec_fn=limit_child, timeout=4, check=False)
            except subprocess.TimeoutExpired:
                return False
            if run.returncode:
                return False
            output.seek(0)
            actual = normalize(output.read(256_000).decode(errors="replace"))
        if kind != "stdin":
            return actual.splitlines()[-1:] == [marker]
        expected = test["expected"]
        return any(actual == normalize(x) for x in (expected if isinstance(expected, list) else [expected]))


if __name__ == "__main__":
    if not os.path.exists("/.dockerenv"):
        raise SystemExit("use the container launcher in grpo_rewards")
    job = json.load(sys.stdin)
    print(json.dumps([verify_one(job["candidate"], job["setup"], test) for test in job["tests"]]))
