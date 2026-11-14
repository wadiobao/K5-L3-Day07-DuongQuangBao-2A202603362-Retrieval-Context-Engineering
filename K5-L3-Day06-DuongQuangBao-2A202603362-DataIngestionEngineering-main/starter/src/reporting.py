"""Incrementally render measured pipeline data into the report template."""
from collections.abc import Mapping
from pathlib import Path
import re


_MARKER = re.compile(r"<!-- AUTO:([a-z0-9_]+) -->")
_DEFAULT_VALUES = {
    "retrieval_results": (
        "| Chưa chạy | — | — | — | — | — | — |"
    ),
    "manual_results": "| Chưa chấm | — | — | N/A | Chưa chấm |",
}


class ReportWriter:
    """Keep report sections up to date as pipeline stages complete."""

    def __init__(
        self,
        template_path: Path,
        output_path: Path,
        initial_values: Mapping[str, str] | None = None,
    ) -> None:
        self.template_path = Path(template_path)
        self.output_path = Path(output_path)
        self._template = self.template_path.read_text(encoding="utf-8")
        self._markers = set(_MARKER.findall(self._template))
        if not self._markers:
            raise ValueError(f"No report markers found in {self.template_path}")
        self._values = {
            marker: _DEFAULT_VALUES.get(
                marker,
                "Chưa có dữ liệu; cập nhật khi bước tương ứng hoàn tất.",
            )
            for marker in self._markers
        }
        self.update(initial_values or {})

    def update(self, values: Mapping[str, str]) -> Path:
        unknown = set(values) - self._markers
        if unknown:
            raise ValueError(f"Unknown report sections: {', '.join(sorted(unknown))}")
        if any(not isinstance(value, str) for value in values.values()):
            raise TypeError("Report section values must be strings")

        self._values.update(values)
        rendered = _MARKER.sub(
            lambda match: self._values[match.group(1)],
            self._template,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(rendered, encoding="utf-8")
        return self.output_path
