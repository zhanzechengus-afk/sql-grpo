import hashlib
import operator
import random
import re


_OPERATIONS = (("+", operator.add), ("-", operator.sub), ("*", operator.mul))
_FINAL_INTEGER = re.compile(r"(?:(?:final\s+)?answer\s*[:=]\s*)?([+-]?)([0-9]+)", re.I)


def _expression(rng, operations):
    if operations == 0:
        value = rng.randint(2, 19)
        return str(value), value
    left_ops = rng.randrange(operations)
    left, first = _expression(rng, left_ops)
    right, second = _expression(rng, operations - left_ops - 1)
    symbol, apply = rng.choice(_OPERATIONS)
    return f"({left} {symbol} {right})", apply(first, second)


def prepare_arithmetic_pool(count, seed):
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < count:
        operations = 3 + len(rows) % 3
        expression, answer = _expression(rng, operations)
        if expression in seen or sum(symbol in expression for symbol, _ in _OPERATIONS) < 2:
            continue
        seen.add(expression)
        rows.append({
            "id": "arithmetic-" + hashlib.sha256(expression.encode()).hexdigest(),
            "prompt": f"Compute {expression}. Give your final answer as a single integer on the last line.",
            "answer": answer,
            "operations": operations,
        })
    return rows


def arithmetic_reward(text, row):
    final_line = text.strip().rsplit("\n", 1)[-1].strip()
    match = _FINAL_INTEGER.fullmatch(final_line.replace("−", "-"))
    if match is None:
        return 0.0
    sign, digits = match.groups()
    digits = digits.lstrip("0") or "0"
    answer = "-" + digits if sign == "-" and digits != "0" else digits
    return float(answer == str(row["answer"]))
