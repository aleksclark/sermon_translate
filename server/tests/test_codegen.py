from __future__ import annotations

from src.codegen import generate_typescript


class TestCodegen:
    def test_generates_nonempty_output(self) -> None:
        ts = generate_typescript()
        assert "AUTO-GENERATED" in ts
        assert "export" in ts

    def test_contains_all_models(self) -> None:
        ts = generate_typescript()
        for name in [
            "SessionStatus",
            "OutputStreamInfo",
            "PipelineInfo",
            "SessionCreate",
            "SessionUpdate",
            "SessionStats",
            "Session",
            "ServerStats",
        ]:
            assert name in ts, f"Missing {name}"

    def test_session_status_is_union(self) -> None:
        ts = generate_typescript()
        assert '"created"' in ts
        assert '"active"' in ts
        assert '"closed"' in ts

    def test_request_models_have_optional_fields(self) -> None:
        ts = generate_typescript()
        lines = ts.split("\n")
        in_create = False
        found_optional = False
        for line in lines:
            if "interface SessionCreate" in line:
                in_create = True
            elif in_create and line.strip() == "}":
                break
            elif in_create and "?" in line:
                found_optional = True
        assert found_optional

    def test_response_models_no_optional(self) -> None:
        ts = generate_typescript()
        lines = ts.split("\n")
        in_server_stats = False
        for line in lines:
            if "interface ServerStats" in line:
                in_server_stats = True
            elif in_server_stats and line.strip() == "}":
                break
            elif in_server_stats and ":" in line:
                assert "?" not in line, f"ServerStats field should not be optional: {line}"
