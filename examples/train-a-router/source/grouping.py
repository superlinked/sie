"""Prospective source-text families; never reads references or model outcomes."""

import hashlib
import json
import re
import unicodedata

VERSION = "bbeh-source-context-families-v1"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def first_sentence(text):
    # Titles such as Prof. and Mr. are not complete story stems.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result = ""
    for part in parts:
        result = (result + " " + part).strip()
        if len(result) >= 35:
            return result
    return result


def context_keys(task, text):
    keys = [("normalized-full-input", normalize(text))]
    covered = False

    def add(kind, value, minimum=30):
        nonlocal covered
        value = normalize(value)
        if len(value) < minimum:
            return
        keys.append((task + ":" + kind, value))
        covered = True

    def match(kind, expression, minimum=30):
        found = re.search(expression, text, re.S | re.I)
        if found:
            add(kind, found.group(1), minimum)

    if task == "linguini":
        # Full prompts keep the original language-problem introduction while
        # varying the requested blank/translation. Tables are an additional
        # link when a setup's paragraph boundary differs.
        add("language-problem-introduction", re.split(r"\n\s*\n", text.strip())[0])
        for block in re.split(r"\n\s*\n", text):
            if "|" in block and len(block) >= 120:
                add("shared-language-table", block)
    elif task == "causal_understanding":
        story = re.sub(r"^\s*Question:\s*", "", text)
        story = re.split(r"\s+Reply Yes or No", story, flags=re.I)[0]
        add("story-opening-sentence", first_sentence(story))
        words = re.findall(r"\w+", normalize(story))
        words = [word for word in words if word not in {"a", "an", "the"}]
        add("story-opening-12-content-tokens", " ".join(words[:12]), 35)
        for sentence in re.split(r"(?<=[.!?])\s+", story):
            if not re.match(r"(?:Reply|If you think people|Did |Was |Is |Would |Does )", sentence, re.I):
                add("shared-story-sentence", sentence, 90)
    elif task == "nycc":
        match("cartoon-description", r"Description:\s*(.*?)\s*Which of the following")
    elif task == "sarc_triples":
        parts = re.findall(r"POST\s+\d+:\s*(.*?)\s*REPLY\s+\d+:\s*(.*?)(?=\s*POST\s+\d+:|\Z)", text, re.S | re.I)
        if len(parts) != 3:
            raise ValueError("SARC source does not contain exactly three post/reply pairs")
        for post, reply in parts:
            add("post-reply-component", post + "\nREPLY:\n" + reply)
    elif task == "word_sorting":
        found = re.search(r"Q:\s*Sort the following words alphabetically:\s*List:\s*(.*?)\s*Thought\s+1:", text, re.S)
        if found:
            add("base-word-multiset", " ".join(sorted(normalize(found.group(1)).split())))
        else:
            found = re.search(
                r"\A(Consider a new alphabet.*?)Sort the following words with the new alphabet and separate them with comma:\s*(.*)\Z",
                text,
                re.S,
            )
            if found:
                add(
                    "custom-alphabet-and-word-multiset",
                    found.group(1) + "\n" + "\n".join(sorted(normalize(found.group(2)).split(","))),
                )
    elif task == "dyck_languages":
        found = re.search(r"\bInput:\s*(.*?)\s*Thought\s+1:", text, re.S)
        if found:
            brackets = re.sub(r"\s+", "", found.group(1))
            if not re.fullmatch(r"[\[\]{}()<>]+", brackets):
                raise ValueError("Unexpected Dyck base-input grammar")
            add("base-bracket-sequence", brackets, 1)
    elif task == "disambiguation_qa":
        match("source-sentence", r"Sentence:\s*(.*?)\s*Which of the following")
    elif task == "sportqa":
        match("main-sport-scenario", r"Main Question:\s*(.*?)(?=\s+A[).])", 40)
    elif task == "time_arithmetic":
        question_numbers = re.findall(r"Question([123]):", text)
        if question_numbers in (["1", "2"], ["1", "2", "3"]):
            # The intervening numerical transformations are part of the
            # composed problem; sharing just one date primitive is not a link.
            add("combined-time-program", text)
    elif task == "buggy_tables":
        match("observed-table", r"\A(.*?)\n\s*\nHowever,", 100)
    elif task == "zebra_puzzles":
        context = re.split(r"\n\s*Question:", text, maxsplit=1)[0]
        lines = [re.sub(r"^Clue\s+\d+:\s*", "", line) for line in context.splitlines() if line.strip()]
        add("unordered-puzzle-clues-and-attributes", "\n".join(sorted(normalize(line) for line in lines)))
    elif task == "hyperbaton":
        add("adjective-order-examples", re.split(r"\n\s*\n", text, maxsplit=1)[0])
    elif task == "multistep_arithmetic":
        # Shared operator definitions alone are a reused primitive, not a problem.
        add("definitions-and-expressions", text)
    elif task == "object_counting":
        match("inventory-before-query", r"\A(.*?)\s+What\s+(?:is|are)\s+")
    elif task == "object_properties":
        match("object-history-before-query", r"\A(.*?)\s+In my current collection,")
    elif task == "temporal_sequence":
        match("calendar-and-constraints", r"\A(.*?)\s+Let X be the length")
    elif task == "web_of_lies":
        match("truth-assignment-context", r"\A(.*?)\b(?:Do|Does)\s+")
    elif task == "geometric_shapes":
        match("svg-path", r"SVG path element:\s*(.*?)(?=\s+(?:Which|What|How|Among|The following))", 50)
    elif task == "spatial_reasoning":
        match("spatial-world-and-path", r"\A(.*?)\s+(?:What (?:will|do|object|is)|Which object)\b", 50)
    elif task == "boardgame_qa":
        match("board-state-rules", r"\A(.*?)\s+Based on the facts, rules, and preferences,")
    elif task == "boolean_expressions":
        options = re.findall(r"(?:\A|\n)\([A-E]\)\s*(.*?)(?=\n\([A-E]\)|\Z)", text, re.S)
        if len(options) == 5:
            add("unordered-expression-options", "\n".join(sorted(normalize(option) for option in options)))
    elif task == "movie_recommendation":
        options = re.findall(r"(?:\A|\n)\([A-J]\)\s*(.*?)(?=\n\([A-J]\)|\Z)", text, re.S)
        if len(options) == 10:
            add("movie-pool", "\n".join(sorted({normalize(part) for option in options for part in option.split(",")})))
    elif task == "shuffled_objects":
        # Literal scenario prefix, excluding the final interrogative clause.
        match("trade-history", r"\A(.*)\s+At the end of\s+")
    else:
        raise ValueError("Unknown BBEH task")
    return keys, covered


def families(rows):
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    first, edges, coverage = {}, [], {}
    for index, row in enumerate(rows):
        keys, covered = context_keys(row["task"], row["request"])
        coverage[row["id"]] = covered
        for kind, value in keys:
            key = (kind, digest(value))
            if key in first:
                previous = first[key]
                join(previous, index)
                edges.append({"left": rows[previous]["id"], "right": row["id"], "rule": kind, "key_sha256": key[1]})
            else:
                first[key] = index
    components = {}
    for index, row in enumerate(rows):
        components.setdefault(find(index), []).append(row["id"])
    identity = {
        member: digest({"version": VERSION, "members": sorted(members)})
        for members in components.values()
        for member in members
    }
    return identity, edges, coverage
