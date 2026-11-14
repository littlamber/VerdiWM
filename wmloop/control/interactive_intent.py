"""Recognize a small, explicit set of round requests in terminal input.

This parser only prepares a plan.  It does not interpret arbitrary research
instructions or grant execution permission.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_NUMBER = r"[+-]?\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+|[A-Za-z]+"
_COUNT = re.compile(rf"(?P<number>{_NUMBER})\s*(?:轮|rounds?\b)", re.IGNORECASE)
_REQUEST = re.compile(
    rf"(?:最多\s*)?(?:跑|运行|执行|run\b|execute\b)\s*(?:最多\s*)?(?P<number>{_NUMBER})\s*(?:轮|rounds?\b)",
    re.IGNORECASE,
)
_SUCCESS = re.compile(r"(?:达到目标|达标)(?:后)?(?:就)?(?:停(?:止)?|结束)|\bstop on success\b", re.IGNORECASE)
_ENGLISH = dict(zip(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
    range(1, 11),
))
_CHINESE = dict(zip("一二三四五六七八九", range(1, 10))) | {"两": 2}


@dataclass(frozen=True)
class RoundIntent:
    rounds: int | None
    goal: str | None
    stop_on_success: bool | None = None


def _round_count(value: str) -> int:
    if re.fullmatch(r"\d+", value):
        result = int(value)
    elif value.casefold() in _ENGLISH:
        result = _ENGLISH[value.casefold()]
    elif value in _CHINESE:
        result = _CHINESE[value]
    elif re.fullmatch(r"[一二三四五六七八九]?十[一二三四五六七八九]?", value):
        tens, units = value.split("十")
        result = _CHINESE.get(tens, 1) * 10 + _CHINESE.get(units, 0)
    else:
        raise ValueError("轮数需要明确的正整数，例如“跑两轮”、‘run two rounds’或 /plan --rounds 2。")
    if result < 1:
        raise ValueError("轮数必须大于 0；例如 /plan --rounds 2。")
    return result


def parse_round_intent(text: str) -> RoundIntent | None:
    """Extract explicit round instructions and preserve the remaining goal.

    Conflicting counts, negation, and alternatives require clarification.  A
    mention such as “提升两轮交互效果” is not a request to run two rounds.
    """

    requests = list(_REQUEST.finditer(text))
    success = _SUCCESS.search(text) is not None
    if not requests and not success:
        return None
    counts = [_round_count(match.group("number")) for match in _COUNT.finditer(text)] if requests else []
    if (counts and len(set(counts)) != 1) or re.search(r"还是|或者|(?:不要|不想|别|不是要?|不用)\s*(?:跑|运行|执行|达标|达到|成功|停)|至少|\b(?:or|not|don't|at least)\b", text, re.IGNORECASE):
        raise ValueError("轮数或停止指令有歧义，未修改目标或计划；请明确使用 --rounds N 和 --[no-]stop-on-success。")
    goal = _SUCCESS.sub("", _REQUEST.sub("", text)).strip(" \t，,。.;；!！")
    # Only remove conversational wrappers; the research objective remains
    # intact, while “我想跑两轮完整研究” preserves the existing objective.
    prefix = re.compile(r"^(?:我想要?|我希望|我们|那就|请(?:帮我)?|帮我|想要?|先|就|只|再|最多|please\b|i want to\b|at most\b|to\b)\s*", re.IGNORECASE)
    while (match := prefix.match(goal)) is not None:
        goal = goal[match.end():].strip(" \t，,。.;；")
    if re.fullmatch(r"(?:完整(?:的)?|整个(?:的)?|研究|实验|闭环|运行|一下|吧|呢|就好|就行|然后|后|就|且|并|停(?:止)?|结束|full|research|experiments?|cycles?|then|and|stop|[,，;；\s])*", goal, re.IGNORECASE):
        goal = ""
    return RoundIntent(rounds=counts[0] if counts else None, goal=goal or None, stop_on_success=True if success else None)
