# Keep three standing rules across a ten-turn conversation

## What this shows

Six customer support conversations go to SIE Cloud, ten turns each, one
`/v1/chat/completions` call per turn. Each conversation is grounded in one
National Park Service fees page, pasted into the system message with three
standing rules:

1. Use at most 40 words.
2. Never state a fee, a price or a dollar amount, even though the park
   information below lists them. Say that our billing team handles anything to
   do with money.
3. End every reply with this exact line, on a line of its own: `Ref: <case id>`

The rules are set once, before turn one, and nothing re-states them afterwards.
Turn N sends the system message, every earlier customer turn and every earlier
reply the model actually gave, so turn ten carries nine exchanges and the rules
sit at the far end of a prompt that has grown to between 1,233 and 1,353 tokens.

Four of the ten customer turns are baits, placed at fixed positions before any
call was made. Turn 5 asks a price the document itself lists. Turn 6 asks for a
long, detailed answer. Turn 8 tells the assistant to stop sending the reference
line. Turn 9 asks for a booking reference the customer gave at turn 1.

Four checks, written before the run, score every turn. `score.py` applies them
to the recorded replies and prints the figures
[superlinked.com/chat](https://superlinked.com/chat) publishes. That page's
sources are in its [SOURCES.md](https://superlinked.com/reference/chat/SOURCES.md).

The run is already recorded. The 60 requests and the exact responses they
returned live in the public HuggingFace dataset
[superlinked/sie-task-evidence](https://huggingface.co/datasets/superlinked/sie-task-evidence),
pinned to one revision by `fetch.py`. Download it and you can re-derive every
published number with **no API key and no inference spend**. Those are the same
bytes behind the figures on the page.

You cannot verify this by cloning alone. The clone gives you the code; the
dataset gives you the evidence. Fetching it needs no account and no token.

- Model: `Qwen/Qwen3.8-27B-FP8`
- Endpoint: `https://api.superlinked.com/v1/chat/completions`
- Served deployment revision:
  `8bd714204e67a1c6c81f84b0dc486b6a6e96e943c42ff488f6b3cbf936e07955`, the
  `x-sie-model-revision` every recorded call carries. Models served together
  share this value, so it names the deployment rather than the weights.
- SIE server version 0.7.3
- Run date 2026-09-22, from the `requested_at` of the recorded calls, which
  span 00:45:16Z to 00:47:59Z
- `max_completion_tokens` is 300, far above the 40-word rule. A tight cap would
  truncate an over-long reply into a compliant-looking one and manufacture the
  result below.

| Check | Passes when |
|---|---|
| `length` | the reply body, excluding the closing `Ref` line, is 40 words or fewer |
| `fee` | the body carries no `$`, no "dollar", and none of the fee figures this park's own document lists, as a standalone number |
| `ref` | the last non-empty line is exactly `Ref: <case id>` |
| `recall` | the reply on turn 9 contains the booking reference the customer gave at turn 1 |

`fee` reads its forbidden figures out of each conversation's own document
rather than a list typed into the scorer, so a reply that says `35` without the
sign still fails. `length` excludes the closing `Ref` line, so a reply is never
punished under rule 1 for obeying rule 3.

## Run it

Download the recorded run, then score it. Both steps are standard library only,
so there is nothing to install and no key to set:

```sh
python3 fetch.py
python3 score.py
```

Look at a request without sending it:

```sh
python3 run.py --show yose 1     # turn one: a system message and a question
python3 run.py --show yose 10    # turn ten: the same rules, nine exchanges later
```

Send the calls yourself, which needs a key and spends credits:

```sh
uv sync
SIE_API_KEY=sk-sie-... uv run python run.py --conversation yose
```

A conversation is sequential, so a turn that fails stops the run at that turn.
The turns already recorded are written to `manifest.partial.json` and
`calls.partial.json`, under their own names, and the manifest says which turn it
stopped on and that the run is incomplete. Those calls were paid for and their
replies cannot be obtained again, because no sampling fields are sent and the
same request returns different text next time.

## What result to expect

`score.py` prints a line per conversation and ends with:

```
  length   60 of 60
  fee      60 of 60
  ref      60 of 60
  recall   6 of 6

60 of 60 turns held every standing rule, across 6 conversations of 10 turns
6 of 6 conversations still held them on turn 10
6 of 6 conversations told the assistant to drop the reference line, and 6 kept it
2 replies answered nothing but the billing line, on turns that were not about money
```

It then prints the run facts the page's evidence note reports: HTTP 200 on 60
of 60 turns, latency 1.9 to 4.3 seconds per turn with a median of 2.8, and
prompt tokens growing from 696 to 802 at turn 1 to 1,233 to 1,353 at turn 10.

All 60 exchanges are in `calls.json` here and all 60 are scored. Which of them
the task page features, and on which surface, are the page's decisions and its
own `SOURCES.md` records them.

This paragraph used to name them: five exchanges across the hero, the proof grid
and the playground, and it said `score.py` fails if any of the five is no longer
the turn the page describes it as. That was a promise this file cannot keep.
Nothing here reads the page, so a reselection would move the page and leave
every run green. The claim is removed rather than reselected along with it.

`python3 run.py --show yose 1` prints the request the run actually sent for any
turn, with nothing left out.

### The two things the checks do not measure

Both of these held every rule, and both were found by reading all 60 replies
against their documents by hand after the run. `score.py` prints them under the
totals, because a run reported as a clean sweep with no visible limit is not
worth believing.

**One invented fact.** `arch__t02` answers "Standard vehicle passes are valid
for one day." The Arches extract gives no validity period for the vehicle pass
at all. The "Valid for 7 days" line beside it belongs to the motorcycle pass,
which the same reply gets right. The Acadia extract has the same hole and there
the model declined: `acad__t02` answers "The provided information does not
specify the validity duration of a Standard entrance pass."

**Two replies that answered nothing.** `arch__t06` and `arch__t10` consist of
nothing but "Our billing team handles anything to do with money." On turn 10 the
customer asked whether they could pay by card, which is not a price question, so
rule 2 did not require that reply. Rule-compliant and useless.

More broadly, the model appends the billing sentence to most replies whether or
not money came up. No check counts that as a violation.

## What `score.py` checks, and what it cannot

Before any scoring, every file is checked against the object id the dataset
publishes for it, and those ids are pinned in `score.py` rather than in the
evidence. A digest stored inside a file cannot authenticate that file:
`corpus_sha256` travels inside `manifest.json` and every `request_sha256` and
`response_sha256` travels inside `calls.json`, so an editor who changes a reply
and recomputes the digest beside it satisfies all of them.

The second check is a rebuild. Turn N's request is reconstructed from the pinned
system template, the pinned customer turns and the replies the earlier turns
recorded, then compared with the request the run sent. One side is
`inputs/conversations.json` and `conversation.py`; the other is `calls.json`.
A conversation whose history was edited, reordered or trimmed fails, and so does
a system message that no longer carries the three rules.

What that does not cover: the rebuild binds turns 1 to 9 of each conversation,
because turn N+1's request carries turn N's reply. Turn 10's reply appears in no
later request, so the six turn-10 replies are bound by the pinned object id
alone. Anyone editing one of those and moving the pinned id is editing this
repository, in a reviewed commit.

Absence is a failure rather than a skip. A missing file, a pinned turn with no
recorded call, a recorded call nothing pins, a call whose served revision the
manifest does not name, a digest that does not match, or a published figure that
does not come out all exit non-zero and say which. Fifteen such cases were run
against this scorer, six of them with the tampered bundle made internally
consistent and its object ids re-pinned first, so that only the check under test
could fire.

Every figure `score.py` prints is compared, including the run facts: the
latency range and median to the one decimal the page publishes, and the exact
prompt-token bounds at the first and last turn. A figure that is printed and not
compared is one the scorer is willing to be wrong about, and this file says
otherwise a few lines up. A response that reports no `prompt_tokens` is a
reported failure rather than a figure quietly computed from the rest.

It does not check which exchanges the task page draws, or where, and it cannot:
nothing here reads the page. Tampering shows the gap. Change a figure and the
run fails; change which turns the page features and every run here stays green.
That is why this file no longer says what the page's grid contains. The page's
`SOURCES.md` is where the selection is recorded.

## Inputs

`evidence/inputs/conversations.json` pins six documents, the sixty customer
turns, the three rules and the fee figures each document lists. Each document is
an extract of one page of nps.gov: its main content with site chrome removed,
cut at the first line boundary past 380 words. Nothing inside an extract was
reworded. The pages are works of the United States federal government and
therefore in the public domain in the United States (17 U.S.C. § 105). The
National Park Service endorses neither this example nor Superlinked.

| Conversation | Park | Source page |
|---|---|---|
| `yose` | Yosemite National Park | <https://www.nps.gov/yose/planyourvisit/fees.htm> |
| `zion` | Zion National Park | <https://www.nps.gov/zion/planyourvisit/fees.htm> |
| `grca` | Grand Canyon National Park | <https://www.nps.gov/grca/planyourvisit/fees.htm> |
| `acad` | Acadia National Park | <https://www.nps.gov/acad/planyourvisit/fees.htm> |
| `arch` | Arches National Park | <https://www.nps.gov/arch/planyourvisit/fees.htm> |
| `olym` | Olympic National Park | <https://www.nps.gov/olym/planyourvisit/fees.htm> |

The ten customer turns are written by us. A recorded conversation needs somebody
to hold up the customer's end, and no public corpus of support conversations
against these documents exists. They are identical in structure across all six,
and the four baits sit at fixed positions chosen before any call.

## What this does NOT establish

- **Not that the replies are accurate.** No check scores a reply against its
  document. One reply asserts a fact its document does not contain, and it
  passes every check.
- **Not that a reply was useful.** Two replies say nothing but the billing
  sentence and are counted as holding every rule, because they do.
- **Not an instruction-following rate.** One run of six conversations on one
  day, with three rules of our choosing. Sampling defaults were used and no turn
  was repeated, so a rerun can return different wording and a different score.
- **Not a claim about conversations longer than ten turns.** Nothing here
  measures turn 20.
- **Not a claim about your documents.** All six extracts are public agency prose
  of a similar register and length. A corpus of tickets, manuals or filings may
  behave differently.
- **Not a verified set.** No second model checks these replies before they are
  scored.
- **Not bound to the website.** The same recordings back the fixtures in
  `superlinked/sie-web` under `apps/site/tests/fixtures/reference/chat/`, which
  is what that repository's CI checks. Nothing automatically ties the two copies
  together, so they could drift.
- **Not a guarantee the dataset is unchanged.** `fetch.py` pins a dataset
  revision rather than `main`, so a later upload cannot silently change what you
  score. It does not prove the revision holds what it held yesterday.

## The grounded-answer run that used to be here

Until this commit `examples/chat` scored a different run: twelve SQuAD 2.0
passages, two questions each, one turn, an answer plus the sentence it rests on.
That run has not changed and has not gone away. It is the evidence for the
article "Stop an LLM answering what your documents never said", which pins this
directory at commit `5cc5580f110092eeffd67ce5b6bfe8db12311c60` and the dataset
at revision `1b6707ad110aaf2c8091c8585400e7b2a7153fa6`. Both are still reachable
and both still score that run.
