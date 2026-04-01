"""Tests for readiness state module."""

from sie_server.core.readiness import is_ready, mark_not_ready, mark_ready


class TestReadinessState:
    """Tests for readiness state management."""

    def test_is_ready_returns_false_initially(self) -> None:
        """is_ready returns False before mark_ready is called."""
        # Reset state
        mark_not_ready()
        assert is_ready() is False

    def test_mark_ready_sets_ready_to_true(self) -> None:
        """mark_ready sets the ready state to True."""
        mark_not_ready()
        assert is_ready() is False

        mark_ready()
        assert is_ready() is True

    def test_mark_not_ready_sets_ready_to_false(self) -> None:
        """mark_not_ready sets the ready state to False."""
        mark_ready()
        assert is_ready() is True

        mark_not_ready()
        assert is_ready() is False

    def test_multiple_mark_ready_calls_are_idempotent(self) -> None:
        """Calling mark_ready multiple times is safe."""
        mark_not_ready()
        mark_ready()
        mark_ready()
        mark_ready()
        assert is_ready() is True

    def test_multiple_mark_not_ready_calls_are_idempotent(self) -> None:
        """Calling mark_not_ready multiple times is safe."""
        mark_ready()
        mark_not_ready()
        mark_not_ready()
        mark_not_ready()
        assert is_ready() is False
