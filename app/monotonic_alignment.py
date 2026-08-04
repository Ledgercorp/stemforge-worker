from __future__ import annotations

from rapidfuzz import fuzz


def _sim(a: str, b: str) -> float:
    return 1.0 if a == b else fuzz.ratio(a, b) / 100.0


def reconcile(expected: list[dict], observed: list[dict]) -> tuple[list[dict], dict]:
    e = [x["normalized"] for x in expected]
    o = [x["normalized"] for x in observed]
    n, m = len(e), len(o)
    method = "monotonic_dynamic_programming_v2"

    if not n:
        return [], {"expected_word_count": 0, "detected_word_count": m,
                    "recognized_or_fuzzy_anchor_count": 0,
                    "anchor_coverage": 0.0, "sequence_similarity": 0.0,
                    "alignment_method": method}

    if not m:
        mapped = [{**x, "start": i * .4, "end": i * .4 + .3,
                   "score": 0.0, "source": "interpolated"}
                  for i, x in enumerate(expected)]
        return mapped, {"expected_word_count": n, "detected_word_count": 0,
                        "recognized_or_fuzzy_anchor_count": 0,
                        "anchor_coverage": 0.0, "sequence_similarity": 0.0,
                        "alignment_method": method}

    egap, ogap, posw = .72, .52, .38
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = dp[i - 1][0] + egap, "E"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = dp[0][j - 1] + ogap, "O"

    for i in range(1, n + 1):
        ef = (i - 1) / max(n - 1, 1)
        for j in range(1, m + 1):
            of = (j - 1) / max(m - 1, 1)
            sub = 1 - _sim(e[i - 1], o[j - 1]) + posw * abs(ef - of)
            cost, action = min((dp[i - 1][j - 1] + sub, "M"),
                               (dp[i - 1][j] + egap, "E"),
                               (dp[i][j - 1] + ogap, "O"))
            dp[i][j], back[i][j] = cost, action

    pairs = []
    i, j = n, m
    while i or j:
        action = back[i][j]
        if action == "M":
            pairs.append((i - 1, j - 1, _sim(e[i - 1], o[j - 1])))
            i, j = i - 1, j - 1
        elif action == "E":
            i -= 1
        elif action == "O":
            j -= 1
        else:
            break
    pairs.reverse()

    mapped: list[dict | None] = [None] * n
    similarities = []
    for ei, oj, similarity in pairs:
        similarities.append(similarity)
        threshold = .76 if len(e[ei]) <= 3 else .66
        if similarity < threshold:
            continue
        src = observed[oj]
        mapped[ei] = {**expected[ei], "start": float(src["start"]),
                      "end": float(src["end"]),
                      "score": min(float(src["score"]), similarity),
                      "source": "recognized_anchor" if similarity >= .999
                                else "fuzzy_anchor"}

    anchors = [i for i, x in enumerate(mapped) if x is not None]
    for i, item in enumerate(expected):
        if mapped[i] is not None:
            continue
        prev = max((a for a in anchors if a < i), default=None)
        nxt = min((a for a in anchors if a > i), default=None)
        if prev is not None and nxt is not None:
            left, right = mapped[prev], mapped[nxt]
            available = max(.05, right["start"] - left["end"])
            step = available / max(nxt - prev, 1)
            start = left["end"] + step * (i - prev - .15)
            end = min(right["start"], start + max(.1, min(.45, step * .72)))
        elif prev is not None:
            start = mapped[prev]["end"] + .06 + .34 * (i - prev - 1)
            end = start + .3
        elif nxt is not None:
            end = max(0.0, mapped[nxt]["start"] - .06 - .34 * (nxt - i - 1))
            start = max(0.0, end - .3)
        else:
            start, end = i * .4, i * .4 + .3
        mapped[i] = {**item, "start": float(max(0.0, start)),
                     "end": float(max(end, start + .05)),
                     "score": 0.0, "source": "interpolated"}

    last = 0.0
    for item in mapped:
        item["start"] = max(float(item["start"]), last)
        item["end"] = max(float(item["end"]), item["start"] + .05)
        last = item["start"]

    anchored = sum(x["source"] != "interpolated" for x in mapped)
    return mapped, {"expected_word_count": n, "detected_word_count": m,
                    "recognized_or_fuzzy_anchor_count": anchored,
                    "anchor_coverage": round(anchored / n, 4),
                    "sequence_similarity": round(sum(similarities) /
                                                   max(len(similarities), 1), 4),
                    "alignment_method": method}


def install() -> None:
    from app import lyric_align
    lyric_align._reconcile = reconcile
