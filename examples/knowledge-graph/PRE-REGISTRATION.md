# Pre-registration

The rules below were fixed before the runs they score, and `score.py` enforces
them rather than reporting whatever it finds.

## The inputs

Ten paragraphs from public filings and recall notices, each pinned in
`inputs/candidates.json` by its SHA-256, with the entity and relation labels
sent with it. Five are displayed on the task page: one hero, three proof
paragraphs and the playground's. The other five are recorded and named, with
the reason each is not shown.

A paragraph's labels are part of its record. `score.py` rebuilds both request
bodies from the pinned text and labels and refuses the run if either does not
reproduce the recorded request, so a paragraph cannot be re-scored under a
schema it was not run with.

## The relation schema asks only what the paragraph can answer

A relation type that a paragraph cannot support is a question with no answer,
and asking it is our defect rather than the model's. Each paragraph's relation
labels name the relations that paragraph can ground.

`score.py` refuses any returned relation whose type is not one of the labels
sent with that paragraph.

## The hand review binds what may be published

A person read every edge the page displays against its source paragraph.
GLiNER2 returns a confidence score and no correctness signal, so a wrong edge
is only ever caught by somebody reading the paragraph.

`score.py` requires a recorded reading for every displayed edge and fails by
name when one is missing. The claim and the check are the same statement: a
displayed edge nobody has read cannot pass.

The review is a superset of the current run. A reading stands as a record of
what was read, so an edge may be absent from a run whose schema no longer asks
for its relation. It may not be absent while that relation is still being sent:
that would be a review of something the run does not support, and `score.py`
fails on it.

## What each published figure means

- **paragraphs recorded**, **shown on the page**: counted from
  `inputs/candidates.json` by `page_role`.
- **edges drawn**: every relation the displayed paragraphs returned. Nothing is
  capped or filtered for display, so the figure is the graph the page draws.

`score.py` fails rather than skipping. It refuses a paragraph whose text does
not match its digest, a candidate with a missing call, a response that does not
match its `response_sha256`, a request the pinned text does not rebuild, a
relations call whose metadata is not the entities call's own output, a
reviewed edge missing while its relation is still being asked for, and a
displayed edge with no recorded reading.
