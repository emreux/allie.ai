"""Earthquakes (AFAD), prayer times (Aladhan by coordinates) and air
quality (Open-Meteo) - D39. Parsing and saying only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any

from allie.store.normalize import normalize_search
from allie.tools.weather import Place

__all__ = [
    "AFAD_EVENTS",
    "AIR_QUALITY",
    "MIN_MAGNITUDE",
    "PRAYERS",
    "QUAKES",
    "QUAKE_HOURS",
    "QUAKE_POOL",
    "Air",
    "Quake",
    "afad_params",
    "air_params",
    "aladhan_params",
    "aladhan_url",
    "aqi_band",
    "describe_air",
    "describe_prayers",
    "describe_quakes",
    "parse_afad",
    "parse_air",
    "parse_aladhan",
]

AFAD_EVENTS = "https://deprem.afad.gov.tr/apiv2/event/filter"
ALADHAN_TIMINGS = "https://api.aladhan.com/v1/timings/{day}"
AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"

QUAKE_HOURS = 24
MIN_MAGNITUDE = 3
QUAKES = 5
# How many are fetched when a province narrows them: a day of M3+ in the
# whole country fits many times over.
QUAKE_POOL = 100

# The six a person asks about, in the day's order. Aladhan's own "Imsak" is
# ten minutes before Fajr by its convention; Diyanet's imsak is Fajr.
PRAYERS = ("Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha")

# The European Air Quality Index bands (EEA), upper bounds.
AQI_BANDS = ((20, "good"), (40, "fair"), (60, "moderate"), (80, "poor"), (100, "very poor"))

_AFAD_TIME = "%Y-%m-%dT%H:%M:%S"


@dataclass(frozen=True, slots=True)
class Quake:
    when: datetime
    magnitude: float
    depth_km: float
    place: str
    area: str


@dataclass(frozen=True, slots=True)
class Air:
    aqi: int | None
    pm25: float | None
    pm10: float | None


def afad_params(now: datetime) -> dict[str, str]:
    """The last `QUAKE_HOURS` hours, in UTC like the dates AFAD answers with
    (checked against Kandilli, 2026-09-27)."""
    end = now.astimezone(UTC)
    start = end - timedelta(hours=QUAKE_HOURS)
    return {
        "start": start.strftime(_AFAD_TIME),
        "end": end.strftime(_AFAD_TIME),
        "minmag": str(MIN_MAGNITUDE),
        "orderby": "timedesc",
        "limit": str(QUAKE_POOL),
    }


def parse_afad(body: Any) -> list[Quake]:
    quakes: list[Quake] = []
    for row in body if isinstance(body, list) else []:
        try:
            quakes.append(
                Quake(
                    when=datetime.strptime(str(row["date"]), _AFAD_TIME).replace(tzinfo=UTC),
                    magnitude=float(row["magnitude"]),
                    depth_km=float(row["depth"]),
                    place=str(row.get("location") or ""),
                    area=" ".join(
                        str(row.get(key) or "") for key in ("province", "district", "location")
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return quakes


def describe_quakes(quakes: list[Quake], *, near: str, tz: tzinfo | None) -> str:
    wanted = normalize_search(near).strip()
    chosen = [quake for quake in quakes if not wanted or wanted in normalize_search(quake.area)]
    where = f" near {near.strip()}" if wanted else ""
    if not chosen:
        return (
            f"No earthquake of magnitude {MIN_MAGNITUDE} or more in the last {QUAKE_HOURS} "
            f"hours{where} (AFAD)."
        )
    lines = [
        f"M{quake.magnitude:.1f} - {quake.when.astimezone(tz):%Y-%m-%d %H:%M} - {quake.place}, "
        f"{quake.depth_km:.0f} km deep"
        for quake in chosen[:QUAKES]
    ]
    return (
        f"Earthquakes of magnitude {MIN_MAGNITUDE} or more in the last {QUAKE_HOURS} "
        f"hours{where} (AFAD), newest first, local time:\n" + "\n".join(lines)
    )


def aladhan_url(day: date) -> str:
    return ALADHAN_TIMINGS.format(day=day.strftime("%d-%m-%Y"))


def aladhan_params(place: Place, method: int) -> dict[str, str | int | float]:
    params: dict[str, str | int | float] = {
        "latitude": place.latitude,
        "longitude": place.longitude,
        "method": method,
    }
    if place.timezone:
        params["timezonestring"] = place.timezone
    return params


def parse_aladhan(body: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    data = body["data"]
    written = str(data["date"]["gregorian"]["date"])  # 27-09-2026
    day = datetime.strptime(written, "%d-%m-%Y").date().isoformat()
    timings = data["timings"]
    # A time may carry its zone in brackets ("05:25 (+03)"); the clock is enough.
    return day, {name: str(timings[name]).split()[0] for name in PRAYERS}


def describe_prayers(label: str, day: str, timings: Mapping[str, str]) -> str:
    listed = ", ".join(f"{name} {timings[name]}" for name in PRAYERS if name in timings)
    return f"Prayer times for {label} on {day}: {listed}. Fajr is the start of the fast (imsak)."


def air_params(place: Place) -> dict[str, str | int | float]:
    return {
        "latitude": place.latitude,
        "longitude": place.longitude,
        "current": "european_aqi,pm2_5,pm10",
        "timezone": "auto",
    }


def parse_air(body: Mapping[str, Any]) -> Air:
    current = body.get("current") or {}
    aqi = current.get("european_aqi")
    return Air(
        aqi=int(aqi) if isinstance(aqi, int | float) else None,
        pm25=_float(current.get("pm2_5")),
        pm10=_float(current.get("pm10")),
    )


def aqi_band(aqi: int) -> str:
    for upper, name in AQI_BANDS:
        if aqi <= upper:
            return name
    return "extremely poor"


def describe_air(label: str, air: Air) -> str:
    if air.aqi is None:
        return f"Open-Meteo has no air quality reading for {label} right now."
    parts = [f"European AQI {air.aqi} ({aqi_band(air.aqi)})"]
    if air.pm25 is not None:
        parts.append(f"PM2.5 {air.pm25:g} µg/m³")
    if air.pm10 is not None:
        parts.append(f"PM10 {air.pm10:g} µg/m³")
    return f"Air quality in {label} now: {', '.join(parts)}."


def _float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
