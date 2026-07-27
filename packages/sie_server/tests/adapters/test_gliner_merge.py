from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.types.responses import Entity


def test_merge_entities_preserves_wider_nested_span() -> None:
    text = "alice@example.com"
    adapter = GLiNERAdapter("test-model")

    result = adapter._merge_entities(
        [
            Entity(text=text, label="EMAIL", score=0.8, start=0, end=len(text)),
            Entity(text="alice", label="EMAIL", score=0.9, start=0, end=5),
        ],
        text,
    )

    assert result == [Entity(text=text, label="EMAIL", score=0.9, start=0, end=len(text))]
