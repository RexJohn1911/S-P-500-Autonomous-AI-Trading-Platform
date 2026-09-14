"""
US Equity Market Calendar and Session Logic.
Provides market hours determination, holiday schedule checks, and multi-symbol data status evaluation.
Handles timezone conversions to America/New_York.
"""

from datetime import date, datetime, time, timezone
from typing import Any, List, Optional
import zoneinfo

# US Eastern Timezone
EASTERN_TZ = zoneinfo.ZoneInfo("America/New_York")

# Regular Trading Hours (US Eastern)
MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)


def _get_easter_sunday(year: int) -> date:
    """Meeus/Jones/Butcher algorithm for Easter Sunday."""
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
    return date(year, month, day)


def get_nyse_holidays(year: int) -> List[date]:
    """
    Compute observed NYSE holidays for a given year.
    - New Year's Day (observed)
    - Martin Luther King Jr. Day (third Monday in January)
    - Washington's Birthday / Presidents' Day (third Monday in February)
    - Good Friday (Friday before Easter Sunday)
    - Memorial Day (last Monday in May)
    - Juneteenth National Independence Day (June 19, observed)
    - Independence Day (July 4, observed)
    - Labor Day (first Monday in September)
    - Thanksgiving Day (fourth Thursday in November)
    - Christmas Day (December 25, observed)
    """
    holidays: List[date] = []

    def observe(holiday_date: date) -> Optional[date]:
        # If Saturday, observed Friday before (except New Year's Day in previous year handled separately)
        if holiday_date.weekday() == 5:
            return date.fromordinal(holiday_date.toordinal() - 1)
        # If Sunday, observed Monday after
        elif holiday_date.weekday() == 6:
            return date.fromordinal(holiday_date.toordinal() + 1)
        return holiday_date

    # 1. New Year's Day
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        holidays.append(date(year, 1, 2))
    elif ny.weekday() != 5:
        holidays.append(ny)

    # 2. MLK Day: 3rd Monday in January
    mondays_jan = [d for d in (date(year, 1, day) for day in range(1, 32)) if d.weekday() == 0]
    if len(mondays_jan) >= 3:
        holidays.append(mondays_jan[2])

    # 3. Washington's Birthday: 3rd Monday in February
    mondays_feb = [d for d in (date(year, 2, day) for day in range(1, 29 if year % 4 != 0 else 30)) if d.weekday() == 0]
    if len(mondays_feb) >= 3:
        holidays.append(mondays_feb[2])

    # 4. Good Friday: 2 days before Easter
    easter = _get_easter_sunday(year)
    good_friday = date.fromordinal(easter.toordinal() - 2)
    holidays.append(good_friday)

    # 5. Memorial Day: last Monday in May
    mondays_may = [d for d in (date(year, 5, day) for day in range(1, 32)) if d.weekday() == 0]
    if mondays_may:
        holidays.append(mondays_may[-1])

    # 6. Juneteenth: June 19 (since 2021/2022)
    if year >= 2021:
        june19 = observe(date(year, 6, 19))
        if june19:
            holidays.append(june19)

    # 7. Independence Day: July 4
    july4 = observe(date(year, 7, 4))
    if july4:
        holidays.append(july4)

    # 8. Labor Day: 1st Monday in September
    mondays_sep = [d for d in (date(year, 9, day) for day in range(1, 31)) if d.weekday() == 0]
    if mondays_sep:
        holidays.append(mondays_sep[0])

    # 9. Thanksgiving Day: 4th Thursday in November
    thursdays_nov = [d for d in (date(year, 11, day) for day in range(1, 31)) if d.weekday() == 3]
    if len(thursdays_nov) >= 4:
        holidays.append(thursdays_nov[3])

    # 10. Christmas Day: Dec 25
    xmas = observe(date(year, 12, 25))
    if xmas:
        holidays.append(xmas)

    return sorted(holidays)


def is_us_equity_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if US Equity markets (NYSE / NASDAQ) are currently in regular trading hours.
    Regular hours: Mon-Fri 09:30 - 16:00 US Eastern, excluding observed holidays.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    eastern_dt = dt.astimezone(EASTERN_TZ)
    # Check weekday (0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday)
    if eastern_dt.weekday() >= 5:
        return False

    # Check holiday
    holidays = get_nyse_holidays(eastern_dt.year)
    if eastern_dt.date() in holidays:
        return False

    # Check time window
    current_time = eastern_dt.time()
    return MARKET_OPEN_TIME <= current_time < MARKET_CLOSE_TIME


def evaluate_market_and_data_status(
    total_symbols: int,
    stale_symbols_count: int,
    current_time: Optional[datetime] = None,
) -> str:
    """
    Determine truthful system market status based on data presence, data freshness,
    and US Equity market session schedule.

    Returns:
    - 'NO_DATA': No symbols or bar data found.
    - 'STALE': All monitored symbols are stale.
    - 'DEGRADED': Partial freshness (some symbols fresh, some stale).
    - 'OPEN': All symbols fresh AND market session is open.
    - 'CLOSED': All symbols fresh AND market session is closed.
    """
    if total_symbols == 0:
        return "NO_DATA"

    if stale_symbols_count == total_symbols:
        return "STALE"

    if stale_symbols_count > 0:
        return "DEGRADED"

    # All symbols are fresh
    if is_us_equity_market_open(current_time):
        return "OPEN"
    return "CLOSED"
