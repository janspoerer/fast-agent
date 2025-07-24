"""
Context truncation manager for LLM conversations.
"""
# region Imports -- Internal Imports
from typing import Any, List, Optional

import tiktoken

# endregion
# region Imports -- External Imports
from mcp_agent.context_dependent import ContextDependent
from mcp_agent.core.agent_types import ContextTruncationMode
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# endregion



class ContextTruncation(ContextDependent):
    """
    Manages the context window of an LLM by truncating the message history
    when it exceeds a specified token limit.

    Takes PromptMessageMultipart and truncates those. It does not deal with provider-specific
    formats. Provider-specific formats should be converted somewhere else!
    """

    def __init__(self, summarizer_llm: Optional[Any] = None):
        """
        Initializes the truncator.
        Args:
            summarizer_llm: An LLM client instance (like your Gemini class)
                            that has a `generate_text(prompt: str) -> str` method.
                            This is only required if you use the SUMMARIZE mode.
        """
        self.logger = get_logger(__name__)
        self._summarization_llm = summarizer_llm
        self.logger.info("Initialized ContextTruncation")

    def truncate_if_required(
        self,
        messages: List[PromptMessageMultipart],
        truncation_mode: Optional[ContextTruncationMode],
        limit: Optional[int],
        model_name: str,
        system_prompt: Optional[str] = None,
    ) -> List[PromptMessageMultipart]:
        """
        Checks if truncation is needed and applies the specified strategy.
        This is a synchronous wrapper for easier integration.
        """
        if not truncation_mode or truncation_mode == ContextTruncationMode.NONE or not limit:
            return messages

        if not self._needs_truncation(messages, limit, model_name, system_prompt):
            return messages

        if truncation_mode == ContextTruncationMode.SUMMARIZE:
            if not self._summarization_llm:
                raise ValueError("Summarizer LLM instance is required for SUMMARIZE mode.")
            return self._summarize_and_truncate(messages, limit, model_name, system_prompt)
        
        elif truncation_mode == ContextTruncationMode.REMOVE:
            return self._truncate(messages, limit, model_name, system_prompt)
        
        return messages

    def _summarize_and_truncate(
        self, messages: List[PromptMessageMultipart], max_tokens: int, model: str, system_prompt: str | None = None
    ) -> List[PromptMessageMultipart]:
        """(Private) Truncates history by summarizing older messages."""
        self.logger.info(f"Context has exceeded {max_tokens} tokens. Applying summarization.")

        system_messages = [m for m in messages if m.role == "system"]
        conversation_messages = [m for m in messages if m.role != "system"]

        # Fallback to simple truncation if conversation is too short
        if len(conversation_messages) <= 2:
            return self._truncate(messages, max_tokens, model, system_prompt)
        
        # Keep the last 2 messages, summarize the rest
        split_index = len(conversation_messages) - 2
        messages_to_summarize = conversation_messages[:split_index]
        messages_to_keep = conversation_messages[split_index:]

        summary_text = self._summarize_messages(messages_to_summarize)
        
        summary_injection = [
            PromptMessageMultipart(
                role="user",
                content=[{"type": "text", "text": f"Here is a summary of our conversation so far: {summary_text}"}]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[{"type": "text", "text": "Thanks, I am caught up. Let's continue."}]
            )
        ]

        new_messages = system_messages + summary_injection + messages_to_keep
        
        # Final check, if still too long, truncate further
        if self._needs_truncation(new_messages, max_tokens, model, system_prompt):
             self.logger.warning("Context still too long after summarization. Applying simple truncation.")
             return self._truncate(new_messages, max_tokens, model, system_prompt)

        return new_messages

    def _summarize_messages(self, messages_to_summarize: List[PromptMessageMultipart]) -> str:
        """Uses the provided LLM to summarize a list of messages."""
        self.logger.info("Summarizing older messages...")
        
        # Format the conversation history into a single string for the prompt
        conversation_text = "\n".join(
            f"{msg.role}: {msg.first_text()}" for msg in messages_to_summarize
        )
        
        prompt = (
            "You are a conversation summarizer. Your task is to create a concise summary "
            "of the following dialogue. The summary should be neutral, retain key information, "
            "and be no more than five sentences long.\n\n"
            f"--- Conversation ---\n{conversation_text}\n\n--- Summary ---"
        )

        # Call the summarizer LLM's text generation method
        response = self._summarization_llm.generate_text(prompt)
        return response.strip()

    def get_summarization_llm(self, model: str):
        """
        Gets an LLM instance for summarization based on the provided model string.
        """
        from mcp_agent.llm.model_factory import create_llm
        self.logger.info(f"Creating a summarization LLM using model: {model}")
        
        # ✅ Create a new LLM instance using the current model string
        # Caching is removed to ensure the correct model is always used.
        return create_llm(
            model=model,
            context=self.context,
            name="summarizer",
        )

    def _estimate_tokens(
        self, messages: List[PromptMessageMultipart], model: str, system_prompt: str | None = None
    ) -> int:
        """Estimate the number of tokens for a list of messages using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.logger.warning(f"Model {model} not found. Using cl100k_base tokenizer.")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = len(encoding.encode(system_prompt)) if system_prompt else 0
        for message in messages:
            # Assuming first_text() gets the main text content
            num_tokens += len(encoding.encode(message.first_text()))
        
        num_tokens += len(messages) * 4 # Approximation for message overhead
        return num_tokens

    def _needs_truncation(
        self, messages: List[PromptMessageMultipart], max_tokens: int, model: str, system_prompt: str | None = None
    ) -> bool:
        """Check if the context needs to be truncated."""
        if not max_tokens:
            return False
        current_tokens = self._estimate_tokens(messages, model, system_prompt)
        return current_tokens > max_tokens

    def _truncate(
        self, messages: List[PromptMessageMultipart], max_tokens: int, model: str, system_prompt: str | None = None
    ) -> List[PromptMessageMultipart]:
        """(Private) Truncates history by removing the oldest messages."""
        initial_tokens = self._estimate_tokens(messages, model, system_prompt)
        self.logger.warning(
            f"Context ({initial_tokens} tokens) has exceeded the limit of {max_tokens} tokens. "
            "Applying simple truncation (remove)."
        )
        
        truncated_messages = list(messages)
        
        # Loop until the token count is within the limit
        while len(truncated_messages) > 1 and self._needs_truncation(
            truncated_messages, max_tokens, model, system_prompt
        ):
            # Find the first non-system message to remove
            for i, msg in enumerate(truncated_messages):
                if msg.role != "system":
                    truncated_messages.pop(i)
                    break 
            else: # No non-system messages left to remove
                break 
        
        final_tokens = self._estimate_tokens(truncated_messages, model, system_prompt)
        self.logger.info(f"Simple truncation complete. New token count: {final_tokens}")

        return truncated_messages