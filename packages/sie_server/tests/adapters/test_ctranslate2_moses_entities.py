from __future__ import annotations

import pytest
from sie_server.adapters.ctranslate2._moses_entities import unescape_moses_entities

_SOURCE = "<2fr> A plain source sentence."


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        ("C &apos; est l &apos; heure d &apos; aller à l &apos; école.", "C'est l'heure d'aller à l'école."),
        ("Je n &apos; ai pas dit qu &apos; Anne s &apos; en va.", "Je n'ai pas dit qu'Anne s'en va."),
        ("Il m &apos; a dit qu &apos; il t &apos; attend.", "Il m'a dit qu'il t'attend."),
        ("J &apos; arrive jusqu &apos; ici.", "J'arrive jusqu'ici."),
        ("Lorsqu &apos; il pleut, quelqu &apos; un reste.", "Lorsqu'il pleut, quelqu'un reste."),
        ("Aujourd &apos; hui, l &apos; été commence.", "Aujourd'hui, l'été commence."),
        ("L &apos; Hôtel ouvre et D &apos; ailleurs ferme.", "L'Hôtel ouvre et D'ailleurs ferme."),
    ],
)
def test_french_elision_rejoins_the_apostrophe(escaped: str, expected: str) -> None:
    assert unescape_moses_entities(escaped, _SOURCE) == expected


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        # Not an elided form: unescaped, spacing kept.
        ("Il dit &apos; oui &apos; .", "Il dit ' oui ' ."),
        # Elided form followed by a digit rather than a word.
        ("l &apos; 2", "l ' 2"),
        # Elided letter inside a longer word.
        ("le mot &apos; x", "le mot ' x"),
    ],
)
def test_other_apostrophes_are_unescaped_without_respacing(escaped: str, expected: str) -> None:
    assert unescape_moses_entities(escaped, _SOURCE) == expected


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        ("Il a dit &quot; bonjour &quot; et il est parti.", 'Il a dit "bonjour" et il est parti.'),
        ("&quot; Oui, tout de suite &quot;, dit-elle.", '"Oui, tout de suite", dit-elle.'),
        ("&quot; Un &quot; puis &quot; deux &quot;.", '"Un" puis "deux".'),
        ("Le projet &quot; Nord &quot; avance.", 'Le projet "Nord" avance.'),
    ],
)
def test_balanced_quote_pairs_attach_to_the_quoted_text(escaped: str, expected: str) -> None:
    assert unescape_moses_entities(escaped, _SOURCE) == expected


@pytest.mark.parametrize(
    ("escaped", "expected"),
    [
        # Odd count: which quote opens is unknown.
        ("Fin du texte. &quot;", 'Fin du texte. "'),
        ("&quot; a &quot; b &quot;", '" a " b "'),
        # A literal quote elsewhere makes the pairing ambiguous.
        ('"Oui" et &quot; non &quot;', '"Oui" et " non "'),
        # A quote that touches a word is not a standalone token.
        ("mot&quot; x &quot;", 'mot" x "'),
        ("&quot; x &quot;mot", '" x "mot'),
        # Not exactly one space inside the pair.
        ("&quot;  x &quot;", '"  x "'),
        ("&quot; &quot;", '" "'),
    ],
)
def test_ambiguous_quotes_are_unescaped_without_respacing(escaped: str, expected: str) -> None:
    assert unescape_moses_entities(escaped, _SOURCE) == expected


def test_remaining_moses_entities_are_unescaped_in_place() -> None:
    escaped = "R &amp; D, a &lt; b &gt; c, &#91; d &#93; et e &#124; f"

    assert unescape_moses_entities(escaped, _SOURCE) == "R & D, a < b > c, [ d ] et e | f"


def test_unescaping_is_a_single_pass() -> None:
    assert unescape_moses_entities("Écrivez &amp;lt; et &amp;apos;", _SOURCE) == "Écrivez &lt; et &apos;"


def test_entities_present_in_the_source_are_kept() -> None:
    source = "<2fr> Tom &amp; Jerry"

    assert unescape_moses_entities("Tom &amp; Jerry", source) == "Tom &amp; Jerry"
    assert unescape_moses_entities("l &apos; ami de Tom &amp; Jerry", source) == "l'ami de Tom &amp; Jerry"


def test_elision_is_only_repaired_for_a_french_target() -> None:
    escaped = "Rock &apos; n &apos; Roll"

    assert unescape_moses_entities(escaped, "<2en> Rock and roll") == "Rock ' n ' Roll"
    assert unescape_moses_entities("l &apos; ami", "<2fr_CA> the friend") == "l'ami"


def test_a_quoted_elided_letter_is_joined_like_an_elision() -> None:
    # Pinned on purpose: a quoted single letter that is an elided form is
    # indistinguishable from an elision, so it joins the word after it.
    escaped = "la lettre &apos; l &apos; est muette"

    assert unescape_moses_entities(escaped, _SOURCE) == "la lettre ' l'est muette"


def test_apostrophe_present_in_the_source_keeps_its_spacing() -> None:
    source = "<2fr> Say &apos; hello"

    assert unescape_moses_entities("Dis l &apos; ami", source) == "Dis l &apos; ami"


def test_text_without_entities_is_returned_unchanged() -> None:
    text = "Une phrase  avec ' des \" signes & et < > [ ] |."

    assert unescape_moses_entities(text, _SOURCE) is text
