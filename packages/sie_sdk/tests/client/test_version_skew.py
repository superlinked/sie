from sie_sdk.client._shared import check_version_skew


class TestVersionSkew:
    def test_same_version_no_warning(self) -> None:
        assert check_version_skew("0.1.6", "0.1.6") is None

    def test_one_minor_apart_no_warning(self) -> None:
        assert check_version_skew("0.1.6", "0.2.0") is None

    def test_two_minor_apart_warns(self) -> None:
        result = check_version_skew("0.1.6", "0.3.0")
        assert result is not None
        assert "behind" in result

    def test_different_major_warns(self) -> None:
        result = check_version_skew("0.1.6", "1.0.0")
        assert result is not None
        assert "major version" in result

    def test_sdk_ahead_warns(self) -> None:
        result = check_version_skew("0.5.0", "0.2.0")
        assert result is not None
        assert "ahead" in result

    def test_invalid_version_no_error(self) -> None:
        assert check_version_skew("invalid", "0.1.0") is None
        assert check_version_skew("0.1.0", "invalid") is None

    def test_empty_version_no_error(self) -> None:
        assert check_version_skew("", "0.1.0") is None
