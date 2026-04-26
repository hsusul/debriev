"""Anthropic verification provider stub."""

from app.services.llm.base import (
    ProviderRequest,
    ProviderResponse,
    VerificationProvider,
    build_placeholder_provider_response,
)


class AnthropicProvider(VerificationProvider):
    """Placeholder Anthropic provider implementation."""

    name = "anthropic"

    def __init__(self, api_key: str, model_version: str) -> None:
        self.api_key = api_key
        self.model_version = model_version

    def verify(self, request: ProviderRequest) -> ProviderResponse:
        return build_placeholder_provider_response(
            request,
            provider_label="Anthropic",
        )
