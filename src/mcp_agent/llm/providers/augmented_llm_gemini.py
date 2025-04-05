import os
from typing import List

from google import genai

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class GeminiAugmentedLLM(AugmentedLLM, AugmentedLLMProtocol):
    def __init__(self, *args, **kwargs) -> None:
        self.provider = "Google (Gemini)"
        self.logger = get_logger(__name__)

        # Now call super().__init__
        super().__init__(*args, **kwargs)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        client = genai.Client(api_key=self._api_key())

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=multipart_messages[-1].first_text()
        )
        return Prompt.assistant(response.text or "")

    def _api_key(self) -> str:
        config = self.context.config
        api_key = None

        if config and config.google:
            api_key = config.google.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "Google (Gemini) API key not configured",
                "The Google (Gemini) API key is required but not set.\n"
                "Add it to your configuration file under google.api_key "
                "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.",
            )
        return api_key
