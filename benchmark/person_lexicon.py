"""Person-mention lexicon and caption normalization of NOBODY-Bench.

`mentions_person(text)` decides whether a caption mentions a person. It flags asserted people and generic agents
alike ("a person is ...", "ready for someone"). `normalize_caption(text)` is applied to model captions before
scoring: it removes a meta-preamble ("Based on the image, here is ...:"), collapses whitespace and strips wrapping
quotes. Both are deterministic and need only the standard library.
"""

from __future__ import annotations

import re

PERSON_RE = re.compile(
    r"\b(man|men|woman|women|person|persons|people|boys?|girls?|child|children|kids?|bab(y|ies)|toddlers?|"
    r"teen(ager)?s?|adults?|guys?|lad(y|ies)|gentlem[ae]n|someone|somebody|anyone|individuals?|players?|"
    r"athletes?|crowds?|famil(y|ies)|couples?|friends?|workers?|riders?|cyclists?|bikers?|surfers?|skiers?|"
    r"skateboarders?|snowboarders?|swimmers?|runners?|joggers?|hikers?|tourists?|pedestrians?|drivers?|"
    r"passengers?|soldiers?|officers?|police(man|men)?|chefs?|cooks?|vendors?|customers?|students?|teachers?|"
    r"audience|spectators?|fans|team(mates)?|mothers?|fathers?|moms?|dads?|parents?|sons?|daughters?|brothers?|"
    r"sisters?|husband|wife|owners?|farmers?|fishermen|fisherman|musicians?|singers?|dancers?|performers?|"
    r"bride|groom|he|she|his|him|herself|himself|humans?|kayakers?|paddlers?|climbers?|golfers?|batters?|"
    r"pitchers?|catchers?|umpires?|referees?|goalies?|shoppers?|diners?|guests?|visitors?|campers?|"
    r"photographers?|cowboys?|rancher|shepherd|herders?|boater|sailors?|fishers?)\b", re.I)


# additional role nouns found in a manual check of 160 model captions
EXTRA_RE = re.compile(
    r"\b(officials?|parachutists?|motorcyclists?|cameram[ae]n|camera operators?|personnel|patients?|caregivers?|"
    r"goalkeepers?|skydivers?|crew(?: members?)?|trainers?|hunters?|staff|nurses?|doctors?|pilots?|waiters?|"
    r"waitress(?:es)?|bartenders?|baristas?|cashiers?|clerks?|shopkeepers?|merchants?|travell?ers?|commuters?|"
    r"residents?|homeowners?|neighbou?rs?|onlookers?|passers?-?by|bystanders?|lifeguards?|firefighters?|"
    r"mechanics?|technicians?|scientists?|researchers?|attendants?|gardeners?|janitors?|bakers?|butchers?|"
    r"carpenters?|laborers?|labourers?|jockeys?|caretakers?|viewers?|worshipp?ers?|monks?|priests?|nuns?|infants?|"
    r"newborns?|grand(?:parents?|mothers?|fathers?|ma|pa|children|child|sons?|daughters?)|uncles?|aunts?|"
    r"boyfriends?|girlfriends?|toddlers?|youngsters?|kayakers?|swimmers?|divers?|"
    r"(?:lone|solitary|human|dark|distant|small|silhouetted|shadowy|single|tiny) figures?|silhouettes? of (?:a|two|three) )\b",
    re.I)

_BASEBALL = re.compile(r"\b(baseball|softball|mound|pitch(es|ed|ing)?|inning|dugout|home plate|batting|bat|umpire|"
                       r"catcher|strike|league|game|field|stadium)\b", re.I)
_HOMOGRAPH = {
    # non-person senses of lexicon words, disambiguated by context (lexicon v2)
    "pitcher": lambda t, s, e: not _BASEBALL.search(t) or re.search(r"\bpitchers? of\b", t[s:e + 4], re.I),
    "pitchers": lambda t, s, e: not _BASEBALL.search(t),
    "batter": lambda t, s, e: not _BASEBALL.search(t),
    "batters": lambda t, s, e: not _BASEBALL.search(t),
    "cook": lambda t, s, e: not re.search(r"\b(a|an|the|his|her|their|head|line|pastry|home|short-order)\s+$",
                                          t[max(0, s - 16):s], re.I),
    "cooks": lambda t, s, e: not re.search(r"\b(the|two|three|four|several|many|some|busy|professional|line|head|"
                                           r"pastry|their|our|other|amateur)\s+$", t[max(0, s - 16):s], re.I),
    "player": lambda t, s, e: bool(re.search(r"\b(music|cd|dvd|mp3|record|video|media|cassette|blu-?ray|vinyl|tape|"
                                             r"portable|audio|disc|game|piano)\s+$", t[max(0, s - 14):s], re.I)),
    "players": lambda t, s, e: bool(re.search(r"\b(music|cd|dvd|mp3|record|video|media|cassette|blu-?ray|vinyl|tape|"
                                              r"audio|disc)\s+$", t[max(0, s - 14):s], re.I)),
    "fans": lambda t, s, e: bool(re.search(r"\b(ceiling|electric|cooling|exhaust|desk|box|hand|table|floor|pedestal|"
                                           r"industrial|wall)\s+$", t[max(0, s - 14):s], re.I)),
}


def _is_person_hit(text: str, m: re.Match) -> bool:
    rule = _HOMOGRAPH.get(m.group(0).lower())
    return not (rule and rule(text, m.start(), m.end()))


def mentions_person(text, version: int = 2) -> bool:
    """True if `text` mentions a person. Version 2 (default, used for every reported number) adds role nouns
    (EXTRA_RE) and rejects non-person senses of homographs by context (water pitcher, cake batter, "ready to cook",
    music player, ceiling fans); version 1 is PERSON_RE alone."""
    text = str(text)
    if version == 1:
        return bool(PERSON_RE.search(text))
    return any(_is_person_hit(text, m) for m in PERSON_RE.finditer(text)) or bool(EXTRA_RE.search(text))


def person_terms(text, version: int = 2) -> list[str]:
    text = str(text)
    hits = [m.group(0).lower() for m in PERSON_RE.finditer(text) if version == 1 or _is_person_hit(text, m)]
    return hits + ([m.group(0).lower() for m in EXTRA_RE.finditer(text)] if version != 1 else [])


def person_spans(text, version: int = 2) -> list[tuple[str, bool]]:
    """`text` cut into consecutive (segment, is_person_term) pieces, for highlighting person words in figures."""
    text = str(text)
    hits = [(m.start(), m.end()) for m in PERSON_RE.finditer(text) if version == 1 or _is_person_hit(text, m)]
    if version != 1:
        hits += [(m.start(), m.end()) for m in EXTRA_RE.finditer(text)]
    out, pos = [], 0
    for a, b in sorted(hits):
        if a < pos:
            continue
        if a > pos:
            out.append((text[pos:a], False))
        out.append((text[a:b], True))
        pos = b
    if pos < len(text):
        out.append((text[pos:], False))
    return out


# ---- caption normalization (applied to model captions before scoring) ----
_PREAMBLE = re.compile(
    r"^\s*(?:based on|here is|here's|from what|looking at|in this (?:image|scene|picture)|the image (?:shows|depicts))"
    r"[^:\n]{0,160}:\s*(?=\S)", re.I)


def has_preamble(text: str) -> bool:
    return bool(_PREAMBLE.match(str(text)))


def normalize_caption(text) -> str:
    t = "" if text is None else str(text)
    t = _PREAMBLE.sub("", t, count=1)
    t = " ".join(t.split()).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t
