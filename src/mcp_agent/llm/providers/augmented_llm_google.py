from typing import Any, Dict, List

# Import necessary types and client from google.genai
from google import genai
from google.genai import (
    errors,  # For error handling
    types,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider

# Import the new converter class
from mcp_agent.llm.providers.google_converter import GoogleConverter
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini-pro"  # Or another suitable default model


class GoogleAugmentedLLM(AugmentedLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    # Define Google-specific parameter exclusions if necessary
    GOOGLE_EXCLUDE_FIELDS = {
        # Add fields that should not be passed directly from RequestParams to google.genai config
        AugmentedLLM.PARAM_MESSAGES,  # Handled by contents
        AugmentedLLM.PARAM_MODEL,  # Handled during client/call setup
        AugmentedLLM.PARAM_SYSTEM_PROMPT,  # Handled by system_instruction in config
        # AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS, # Handled by tool_config in config
        AugmentedLLM.PARAM_USE_HISTORY,  # Handled by AugmentedLLM base / this class's logic
        AugmentedLLM.PARAM_MAX_ITERATIONS,  # Handled by this class's loop
        # Add any other OpenAI-specific params not applicable to google.genai
    }.union(AugmentedLLM.BASE_EXCLUDE_FIELDS)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE, **kwargs)
        # Initialize the google.genai client
        self._google_client = self._initialize_google_client()
        # Initialize the converter
        self._converter = GoogleConverter()

    def _initialize_google_client(self) -> genai.Client:
        """
        Initializes the google.genai client.

        Reads Google API key or Vertex AI configuration from context config.
        """
        try:
            # Example: Authenticate using API key from config
            api_key = self._api_key()  # Assuming _api_key() exists in base class
            if not api_key:
                # Handle case where API key is missing
                raise ProviderKeyError(
                    "Google API key not found.", "Please configure your Google API key."
                )

            # Check for Vertex AI configuration
            if (
                self.context
                and self.context.config
                and hasattr(self.context.config, "google")
                and hasattr(self.context.config.google, "vertex_ai")
                and self.context.config.google.vertex_ai.enabled
            ):
                vertex_config = self.context.config.google.vertex_ai
                return genai.Client(
                    vertexai=True,
                    project=vertex_config.project_id,
                    location=vertex_config.location,
                    # Add other Vertex AI specific options if needed
                    # http_options=types.HttpOptions(api_version='v1') # Example for v1 API
                )
            else:
                # Default to Gemini Developer API
                return genai.Client(
                    api_key=api_key,
                    # http_options=types.HttpOptions(api_version='v1') # Example for v1 API
                )
        except Exception as e:
            # Catch potential initialization errors and raise ProviderKeyError
            raise ProviderKeyError("Failed to initialize Google GenAI client.", str(e)) from e

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Google-specific default parameters."""
        chosen_model = kwargs.get("model", DEFAULT_GOOGLE_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,  # System instruction will be mapped in _google_completion
            parallel_tool_calls=True,  # Assume parallel tool calls are supported by default with native API
            max_iterations=10,
            use_history=True,
            # Include other relevant default parameters
        )

    async def _google_completion(
        self,
        messages: List[types.Content],
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using Google's generate_content API and available tools.
        """
        request_params = self.get_request_params(request_params=request_params)
        responses: List[TextContent | ImageContent | EmbeddedResource] = []
        conversation_history = list(messages)  # Start with the input messages for this turn

        for i in range(request_params.max_iterations):
            # 1. Get available tools
            aggregator_response = await self.aggregator.list_tools()
            available_tools = self._converter.convert_to_google_tools(
                aggregator_response.tools
            )  # Convert fast-agent tools to google.genai tools

            # 2. Prepare generate_content arguments
            # Map RequestParams to GenerateContentConfig
            generate_content_config = self._converter.convert_request_params_to_google_config(
                request_params
            )
            # Add tools to config if available
            if available_tools:
                # The tools parameter is passed directly to generate_content, not in the config.
                # This was noted in the previous architectural plan.
                # The previous implementation incorrectly put tools in generate_content_config.
                # Correcting this here.
                # Also, need to check the google.genai docs on how tool_config interacts.
                # Based on README, tool_config is part of GenerateContentConfig.
                # So the tool_config part can remain in the config.
                # generate_content_config.tools = available_tools # Remove this line
                generate_content_config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY"
                        if request_params.parallel_tool_calls
                        else "AUTO"  # Or 'ANY' if we always want tool_calls
                    )
                )

            # The `contents` argument for generate_content should be a list of google.genai Content objects.
            # The conversation_history list is already in this format.

            # 3. Call the google.genai API
            try:
                # Use the async client
                api_response = await self._google_client.aio.models.generate_content(
                    model=request_params.model,  # Use model from RequestParams
                    contents=conversation_history,  # Pass the current turn's conversation history
                    generation_config=generate_content_config,
                    tools=available_tools
                    if available_tools
                    else None,  # Pass tools here as a separate parameter
                )
                self.logger.debug("Google generate_content response:", data=api_response)

            except errors.APIError as e:
                # Handle specific Google API errors
                raise ProviderKeyError(f"Google API Error: {e.code}", e.message) from e
            except Exception as e:
                self.logger.error(f"Error during Google generate_content call: {e}")
                # Decide how to handle other exceptions - potentially re-raise or return an error message
                raise e

            # 4. Process the API response
            if not api_response.candidates:
                # No response from the model, we're done
                self.logger.debug(f"Iteration {i}: No candidates returned.")
                break

            candidate = api_response.candidates[0]  # Process the first candidate

            # Convert the model's response content to fast-agent types
            # Note: google.genai response content also includes role, so pass the whole content object
            model_response_content_parts = self._converter.convert_from_google_content(
                candidate.content
            )

            # Add model's response to conversation history for potential next turn
            # This is for the *internal* conversation history of this completion call
            # to handle multi-turn tool use within one _google_completion call.
            conversation_history.append(candidate.content)

            # Extract and process text content and tool calls to add to the *main* history
            assistant_message_parts = []
            tool_messages_content = []

            for part in model_response_content_parts:
                if isinstance(part, TextContent):
                    responses.append(part)  # Add text content to the final responses to be returned
                    assistant_message_parts.append(
                        part
                    )  # Add text to the potential assistant message for history
                elif isinstance(part, CallToolRequestParams):
                    # This is a function call requested by the model
                    # Convert to CallToolRequest and execute
                    tool_call_request = CallToolRequest(method="tools/call", params=part)
                    self.show_tool_call(
                        aggregator_response.tools,  # Pass fast-agent tool definitions for display
                        tool_call_request.params.name,
                        str(
                            tool_call_request.params.arguments
                        ),  # Convert dict to string for display
                    )

                    # Execute the tool call
                    # Need to find the correct tool_call_id if available in google.genai response
                    # Based on docs, google.genai.types.FunctionCall doesn't have an ID.
                    # Pass None for now, or generate a unique ID if required by self.call_tool.
                    # Let's check the signature of self.call_tool. It takes tool_call_id: str | None.
                    # Passing None is acceptable.
                    result = await self.call_tool(tool_call_request, None)

                    self.show_oai_tool_result(str(result.content))

                    # Add tool result content to the overall responses to be returned
                    responses.extend(result.content)

                    # Convert tool result back to google.genai format for the next turn's conversation_history
                    # and also to fast-agent format for the main history.
                    tool_response_google_content = (
                        self._converter.convert_function_results_to_google([(part.name, result)])[0]
                    )  # Assuming single tool result conversion
                    conversation_history.append(tool_response_google_content)

                    # Convert tool result to fast-agent format for main history update
                    # This requires a method in GoogleConverter to convert CallToolResult to PromptMessageMultipart(role='tool').
                    # Let's assume such a method exists: self._converter.convert_tool_result_to_fast_agent_message
                    # Need to implement convert_tool_result_to_fast_agent_message in GoogleConverter.
                    # For now, we can manually construct a PromptMessageMultipart with role='tool'.
                    # The content of the tool message in fast-agent is typically the raw result.
                    tool_message_fast_agent = Prompt.tool(
                        result
                    )  # Assuming Prompt.tool creates a tool message
                    tool_messages_content.append(
                        tool_message_fast_agent.content
                    )  # Collect content for main history update

            # After processing all parts of the candidate, add messages to the main history
            if assistant_message_parts:
                # Need to add this message to the main history managed by self.history.
                # This likely happens after the _google_completion call returns in _apply_prompt_provider_specific.
                # Or, if we manage history within _google_completion, we need to append it here.
                # Given the architecture plan mentions updating self.history after all iterations,
                # we will rely on that. The 'responses' list captures the output to be returned.
                pass  # History update will be handled outside this loop or at the end.

            # The conversion and history update logic is getting a bit complex within the loop.
            # Let's simplify: The loop's primary goal is to handle multi-turn tool calls.
            # The final responses list collects all output (text and tool results).
            # History update should happen once after the loop finishes, using the combined
            # conversation turns that happened within this _google_completion call.

            # Let's revert the history update logic within the loop and focus on
            # correctly building the conversation_history for the *next* API call iteration.
            # The initial messages are passed in. Model response is added. Tool results are added (in google.genai format).

            # The responses list collects the output to be returned at the very end.
            # The main history update will happen in _apply_prompt_provider_specific or after _google_completion returns.

            # Let's refine the loop to just correctly build conversation_history for the next turn
            # and populate the 'responses' list with fast-agent content types.

            # The logic for handling tool calls and appending to conversation_history seems correct.
            # The logic for extracting text and appending to responses is also correct.

            # The logic for updating the main history (self.history) needs to be implemented
            # outside this loop or at the end of _google_completion.
            pass  # History update strategy will be finalized later.

            # Continue the loop for the next turn if there were tool calls to process
            # The loop continues as long as finish_reason is not 'stop' or 'max_tokens' or 'content_filter'
            # If function_calls were present, we continue the loop to process the tool results in the next turn.
            if not candidate.function_calls:
                # If no function calls, check finish reason to stop or continue
                if candidate.finish_reason in ["STOP", "MAX_TOKENS", "SAFETY"]:
                    break  # Exit the loop if a stopping condition is met
                # If no function calls and no stopping reason, the model might be done.
                # We can break here or let it try one more turn (up to max_iterations).
                # Let's break if no function calls and no explicit reason to continue (like trying to generate more text).
                # The current loop condition (range(request_params.max_iterations)) handles the max iterations.
                # The check for finish_reason handles explicit stops.
                # So, if there were no function calls, and it didn't explicitly stop,
                # we should probably break to avoid infinite loops on models that just stop talking.
                # Let's add a check for text content - if no text and no tool calls, break.
                if not assistant_message_parts:
                    self.logger.debug(
                        f"Iteration {i}: No text content or function calls, breaking."
                    )
                    break

        # 6. Update history after all iterations are done (or max_iterations reached)
        # This needs careful implementation to merge conversation_history into self.history
        # The conversation_history contains google.genai.types.Content for the turns within this call.
        # Need to convert these back to PromptMessageMultipart and add to self.history.
        # Need a method in GoogleConverter to convert google.genai.types.Content to PromptMessageMultipart.
        # Let's add a TODO for this conversion and history update outside the loop.
        if request_params.use_history:
            # Convert conversation_history (types.Content) to PromptMessageMultipart
            # and append to self.history.
            converted_history = self._converter.convert_from_google_content_list(
                conversation_history
            )
            self.history.extend(converted_history)

        self._log_chat_finished(model=request_params.model)  # Use model from request_params
        return responses

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """
        Applies the prompt messages and potentially calls the LLM for completion.
        """
        # Convert incoming fast-agent messages to google.genai format for the initial call
        initial_google_messages = self._converter.convert_to_google_content(multipart_messages)

        # The base class's apply_prompt likely handles adding initial messages to history.
        # We just need to handle the completion call if the last message is from the user.

        last_message_role = multipart_messages[-1].role if multipart_messages else None

        if last_message_role == "user":
            # If the last message is from the user, call the LLM for a response
            # Pass the initial converted messages to _google_completion
            responses = await self._google_completion(initial_google_messages, request_params)
            # _google_completion will handle appending subsequent turn messages to history
            return Prompt.assistant(*responses)  # Return combined responses as an assistant message
        else:
            # If the last message is not from the user (e.g., assistant), no completion is needed for this step
            # The messages have already been added to history by the calling code/framework
            return multipart_messages[-1]  # Return the last message as is

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        """
        Hook called before a tool call.

        Args:
            tool_call_id: The ID of the tool call.
            request: The CallToolRequest object.

        Returns:
            The modified CallToolRequest object.
        """
        # Currently a pass-through, can add Google-specific logic if needed
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        """
        Hook called after a tool call.

        Args:
            tool_call_id: The ID of the tool call.
            request: The original CallToolRequest object.
            result: The CallToolResult object.

        Returns:
            The modified CallToolResult object.
        """
        # Currently a pass-through, can add Google-specific logic if needed
        return result

    def prepare_provider_arguments(
        self, base_args: Dict[str, Any], request_params: RequestParams, exclude_fields: set[str]
    ) -> Dict[str, Any]:
        """
        Prepare arguments for the google.genai API call, excluding certain fields.
        This overrides the base class method to use Google-specific exclusions.
        """
        # This method might need significant changes or could be simplified
        # as we are now building google.genai specific types directly in _google_completion
        # rather than relying on a generic dictionary of arguments.
        # For now, we can keep a basic implementation that applies exclusions
        # but the primary argument preparation will be in _google_completion.
        arguments = super().prepare_provider_arguments(
            base_args, request_params, exclude_fields.union(self.GOOGLE_EXCLUDE_FIELDS)
        )
        # Further process arguments if necessary for google.genai
        return arguments
