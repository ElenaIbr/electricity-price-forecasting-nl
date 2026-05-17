"""NL holiday calendar.

Минимальный hand-coded набор: New Year, King's Day, Liberation Day, Christmas
+ Easter-relative (Good Friday, Easter Monday, Ascension, Whit Monday).

Для thesis-уровня этого достаточно. Для production стоит подключить
`python-holidays` (`pip install holidays`) — но тогда замена должна
идти под версионированием feature_eng_hash.
"""
from __future__ import annotations

import datetime as dt


_FIXED_DATES = [
    (1, 1),    # New Year's Day
    (4, 27),   # King's Day
    (5, 5),    # Liberation Day (5-yearly bank holiday, but kept for stability)
    (12, 25),  # Christmas
    (12, 26),  # Boxing Day
]

# Easter Sunday (computed via Anonymous Gregorian algorithm)
def _easter_sunday(year: int) -> dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def nl_holidays(years: range | list[int]) -> set[dt.date]:
    """Возвращает set NL holidays для указанных лет."""
    out: set[dt.date] = set()
    for y in years:
        for m, d in _FIXED_DATES:
            out.add(dt.date(y, m, d))
        easter = _easter_sunday(y)
        out.add(easter - dt.timedelta(days=2))   # Good Friday
        out.add(easter + dt.timedelta(days=1))   # Easter Monday
        out.add(easter + dt.timedelta(days=39))  # Ascension Day
        out.add(easter + dt.timedelta(days=50))  # Whit Monday
    return out
