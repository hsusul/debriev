"""Parser registry for source-type-specific parsing strategies."""

from collections.abc import Iterable

from app.core.enums import SourceType
from app.core.exceptions import ValidationError
from app.services.parsing.base import SourceParser


class ParserRegistry:
    """Explicit source-type to parser mapping."""

    def __init__(self, parsers: Iterable[SourceParser] | None = None) -> None:
        self._parsers: dict[SourceType, SourceParser] = {}
        for parser in parsers or []:
            self.register(parser)

    def register(self, parser: SourceParser) -> None:
        self._parsers[parser.source_type] = parser

    def get(self, source_type: SourceType) -> SourceParser:
        parser = self._parsers.get(source_type)
        if parser is None:
            raise ValidationError(f"No parser is registered for source type {source_type.value}.")
        return parser


def build_default_parser_registry() -> ParserRegistry:
    from app.services.parsing.transcript_parser import DeclarationParser, ExhibitParser, TranscriptParser

    return ParserRegistry(
        parsers=[
            TranscriptParser(),
            DeclarationParser(),
            ExhibitParser(),
        ]
    )
