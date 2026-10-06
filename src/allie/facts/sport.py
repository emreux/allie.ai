"""Formula 1 (Jolpica, the Ergast successor) and football (ESPN's public
site API) - D39. TheSportsDB's free key was measured in N0 and dropped: five
of a round's nine matches, the table's top five, the wrong team on a search.
ESPN's API is undocumented; when it changes, the tool says it did not answer
and the model looks it up instead."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any

from allie.store.normalize import normalize_search

__all__ = [
    "ESPN_SITE",
    "ESPN_STANDINGS",
    "F1_LAST",
    "F1_NEXT",
    "Match",
    "Standing",
    "Team",
    "describe_last_race",
    "describe_next_race",
    "describe_round",
    "describe_team",
    "find_team",
    "parse_matches",
    "parse_table",
    "parse_teams",
]

F1_NEXT = "https://api.jolpi.ca/ergast/f1/current/next.json"
F1_LAST = "https://api.jolpi.ca/ergast/f1/current/last/results.json"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/soccer/{league}/standings"

# The sessions of a race weekend, in the order they run; the race is last.
SESSIONS = (
    ("FirstPractice", "First practice"),
    ("SecondPractice", "Second practice"),
    ("ThirdPractice", "Third practice"),
    ("SprintQualifying", "Sprint qualifying"),
    ("Sprint", "Sprint"),
    ("Qualifying", "Qualifying"),
)
PODIUM = 3
PLAYED = 3
UPCOMING = 2
TOP = 5


@dataclass(frozen=True, slots=True)
class Team:
    id: str
    name: str
    names: tuple[str, ...]  # every spelling ESPN gives, folded


@dataclass(frozen=True, slots=True)
class Match:
    when: datetime
    home: str
    away: str
    home_score: str | None
    away_score: str | None
    state: str  # "pre", "in" or "post"


@dataclass(frozen=True, slots=True)
class Standing:
    rank: int
    team: str
    team_id: str
    points: int
    played: int


# -- Formula 1 -------------------------------------------------------------------


def describe_next_race(body: Mapping[str, Any], *, tz: tzinfo | None) -> str:
    races = body["MRData"]["RaceTable"]["Races"]
    if not races:
        return "No Formula 1 race is scheduled for the rest of this season."
    race = races[0]
    circuit = race["Circuit"]
    location = circuit.get("Location") or {}
    where = ", ".join(str(location[key]) for key in ("locality", "country") if location.get(key))
    times = [
        f"{label} {_moment(race[key], tz)}"
        for key, label in SESSIONS
        if isinstance(race.get(key), dict)
    ]
    times.append(f"Race {_moment(race, tz)}")
    return (
        f"Next: {race['raceName']}, round {race['round']}, {circuit['circuitName']} ({where}). "
        f"Local times: {', '.join(times)}."
    )


def describe_last_race(body: Mapping[str, Any]) -> str:
    races = body["MRData"]["RaceTable"]["Races"]
    if not races:
        return "No Formula 1 race has been run this season yet."
    race = races[0]
    podium = ", ".join(
        f"{result['position']}. {result['Driver']['givenName']} {result['Driver']['familyName']} "
        f"({result['Constructor']['name']})"
        for result in race["Results"][:PODIUM]
    )
    return f"Last: {race['raceName']} ({race['date']}): {podium}."


def _moment(entry: Mapping[str, Any], tz: tzinfo | None) -> str:
    when = datetime.fromisoformat(f"{entry['date']}T{entry.get('time', '00:00:00Z')}")
    return f"{when.astimezone(tz):%a %Y-%m-%d %H:%M}"


# -- football ----------------------------------------------------------------------


def parse_teams(body: Mapping[str, Any]) -> list[Team]:
    teams: list[Team] = []
    for entry in body["sports"][0]["leagues"][0]["teams"]:
        team = entry["team"]
        spellings = (
            team.get(key)
            for key in ("displayName", "shortDisplayName", "name", "location", "abbreviation")
        )
        names = tuple(
            dict.fromkeys(normalize_search(str(name)).strip() for name in spellings if name)
        )
        teams.append(Team(id=str(team["id"]), name=str(team["displayName"]), names=names))
    return teams


def find_team(teams: list[Team], spoken: str) -> Team | None:
    """The team the user named: a spelling equal to theirs first, then one
    that holds theirs or is held by it ("Rizespor" is "Caykur Rizespor")."""
    wanted = normalize_search(spoken).strip()
    if not wanted:
        return None
    for team in teams:
        if wanted in team.names:
            return team
    for team in teams:
        # Longer than an abbreviation: "ala" is in "galatasaray".
        if any(len(name) > 3 and (wanted in name or name in wanted) for name in team.names):
            return team
    return None


def parse_matches(body: Mapping[str, Any]) -> list[Match]:
    matches: list[Match] = []
    for event in body.get("events") or []:
        competition = event["competitions"][0]
        sides = {side["homeAway"]: side for side in competition["competitors"]}
        status = competition.get("status") or event.get("status") or {}
        matches.append(
            Match(
                when=datetime.fromisoformat(str(event["date"])),
                home=str(sides["home"]["team"]["displayName"]),
                away=str(sides["away"]["team"]["displayName"]),
                home_score=_score(sides["home"].get("score")),
                away_score=_score(sides["away"].get("score")),
                state=str(status.get("type", {}).get("state", "")),
            )
        )
    return matches


def parse_table(body: Mapping[str, Any]) -> list[Standing]:
    rows: list[Standing] = []
    for index, entry in enumerate(body["children"][0]["standings"]["entries"], start=1):
        stats = {stat["name"]: stat.get("value") for stat in entry["stats"]}
        rows.append(
            Standing(
                rank=int(stats.get("rank") or index),
                team=str(entry["team"]["displayName"]),
                team_id=str(entry["team"]["id"]),
                points=int(stats.get("points") or 0),
                played=int(stats.get("gamesPlayed") or 0),
            )
        )
    return sorted(rows, key=lambda row: row.rank)


def describe_team(
    team: Team,
    played: list[Match],
    upcoming: list[Match],
    table: list[Standing],
    *,
    tz: tzinfo | None,
) -> str:
    lines: list[str] = []
    standing = next((row for row in table if row.team_id == team.id), None)
    if standing is not None:
        lines.append(
            f"{team.name}: {_ordinal(standing.rank)} in the table with {standing.points} points "
            f"after {standing.played} games."
        )
    else:
        lines.append(f"{team.name}:")
    done = sorted((m for m in played if m.state == "post"), key=lambda m: m.when, reverse=True)
    if done:
        lines.append("Last results:")
        lines.extend(_line(match, tz) for match in done[:PLAYED])
    ahead = sorted((m for m in upcoming if m.state == "pre"), key=lambda m: m.when)
    if ahead:
        lines.append("Next matches:")
        lines.extend(_line(match, tz) for match in ahead[:UPCOMING])
    return "\n".join(lines)


def describe_round(matches: list[Match], table: list[Standing], *, tz: tzinfo | None) -> str:
    lines = ["The latest round, local time:"]
    lines.extend(_line(match, tz) for match in sorted(matches, key=lambda m: m.when))
    if table:
        top = ", ".join(f"{row.rank}. {row.team} {row.points}" for row in table[:TOP])
        lines.append(f"Top of the table: {top}.")
    return "\n".join(lines)


def _line(match: Match, tz: tzinfo | None) -> str:
    when = f"{match.when.astimezone(tz):%Y-%m-%d %H:%M}"
    if match.home_score is None or match.away_score is None or match.state == "pre":
        return f"{when} {match.home} - {match.away}"
    live = " (playing now)" if match.state == "in" else ""
    return f"{when} {match.home} {match.home_score}-{match.away_score} {match.away}{live}"


def _score(value: object) -> str | None:
    if isinstance(value, dict):
        value = value.get("displayValue")
    return str(value) if value not in (None, "") else None


def _ordinal(rank: int) -> str:
    suffix = "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix}"
