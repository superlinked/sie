from sie_server.api.helpers import SERVER_VERSION_HEADER, ResponseBuilder


class TestVersionHeader:
    def test_server_version_header_in_response(self) -> None:
        headers = ResponseBuilder.build_headers()
        assert SERVER_VERSION_HEADER in headers
        assert headers[SERVER_VERSION_HEADER] != ""
        assert headers[SERVER_VERSION_HEADER] != "unknown"
