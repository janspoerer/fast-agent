from typing import Any, Dict, List, Tuple

# Import necessary types from google.genai
from google.genai import types
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.tools.tool_definition import ToolDefinition


class GoogleConverter:
    """
    Converts between fast-agent and google.genai data structures.
    """

    def _clean_schema_for_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively removes 'additionalProperties' from a schema dictionary."""
        cleaned_schema = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue  # Skip this key
            if isinstance(value, dict):
                cleaned_schema[key] = self._clean_schema_for_google(value)
            elif isinstance(value, list):
                cleaned_schema[key] = [
                    self._clean_schema_for_google(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned_schema[key] = value
        return cleaned_schema

    def convert_to_google_content(
        self, messages: List[PromptMessageMultipart]
    ) -> List[types.Content]:
        """
        Converts a list of fast-agent PromptMessageMultipart to google.genai types.Content.
        Handles different roles and content types (text, images, etc.).
        """
        google_contents: List[types.Content] = []
        for message in messages:
            parts: List[types.Part] = []
            for part in message.content:
                if isinstance(part, TextContent):
                    parts.append(types.Part.from_text(text=part.text))
                elif isinstance(part, ImageContent):
                    # Assuming ImageContent has a 'data' attribute with bytes or a 'url' attribute
                    # The google.genai library supports image bytes or GCS URIs.
                    if hasattr(part, "data") and part.data:
                        # Use the mime type provided in ImageContent
                        parts.append(types.Part.from_bytes(data=part.data, mime_type=part.mimeType))
                    elif hasattr(part, "url") and part.url:
                        # Assumes the URL is a GCS URI if using Vertex AI, or needs file upload for Gemini API
                        # Handling file uploads adds complexity and might be a separate step/consideration
                        # For simplicity initially, might only support text and tool_code
                        raise NotImplementedError(
                            "ImageContent from URL not yet supported for google.genai conversion."
                        )
                # Add other content types if needed (e.g., EmbeddedResource)
            if parts:
                # Map fast-agent roles to google.genai roles
                # fast-agent: 'user', 'assistant', 'tool', 'system'
                # google.genai: 'user', 'model', 'tool'
                # System instructions are handled separately in GenerateContentConfig
                google_role = (
                    "user"
                    if message.role == "user"
                    else ("model" if message.role == "assistant" else "tool")
                )  # Mapping tool to tool role

                google_contents.append(types.Content(role=google_role, parts=parts))

        return google_contents

    def convert_to_google_tools(self, tools: List[ToolDefinition]) -> List[types.Tool]:
        """
        Converts a list of fast-agent ToolDefinition to google.genai types.Tool.
        """
        google_tools: List[types.Tool] = []
        for tool in tools:
            # Assuming ToolDefinition.inputSchema is a dict representing a JSON schema
            # Clean the schema to remove 'additionalProperties' which google.genai.types.Schema might not support
            cleaned_input_schema = self._clean_schema_for_google(tool.inputSchema)

            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description if tool.description else "",
                parameters=types.Schema(
                    **cleaned_input_schema
                ),  # Convert cleaned JSON schema dict to google.genai Schema
            )
            google_tools.append(types.Tool(function_declarations=[function_declaration]))
        return google_tools

    def convert_from_google_content(
        self, content: types.Content
    ) -> List[TextContent | ImageContent | EmbeddedResource | CallToolRequestParams]:
        """
        Converts google.genai types.Content from a model response to a list of
        fast-agent content types or tool call requests.
        """
        fast_agent_parts: List[
            TextContent | ImageContent | EmbeddedResource | CallToolRequestParams
        ] = []
        for part in content.parts:
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_call:
                # This part converts a function_call from the model's response
                # into a CallToolRequestParams object for fast-agent's tool execution.
                fast_agent_parts.append(
                    CallToolRequestParams(
                        name=part.function_call.name,
                        arguments=part.function_call.args,  # args is already a dict
                    )
                )
            # Add handling for other part types like file_data (images), etc.
            # This might require fetching data from URIs if applicable.
        return fast_agent_parts

    def convert_from_google_function_call(
        self, function_call: types.FunctionCall
    ) -> CallToolRequest:
        """
        Converts a single google.genai types.FunctionCall to a fast-agent CallToolRequest.
        """
        return CallToolRequest(
            method="tools/call",  # Standard method for tool calls in fast-agent
            params=CallToolRequestParams(
                name=function_call.name,
                arguments=function_call.args,  # args is already a dict
            ),
        )

    def convert_function_results_to_google(
        self, tool_results: List[Tuple[str, CallToolResult]]
    ) -> List[types.Content]:
        """
        Converts a list of fast-agent tool results to google.genai types.Content
        with role 'tool'.
        """
        google_tool_response_contents: List[types.Content] = []
        for tool_name, tool_result in tool_results:
            # Assuming CallToolResult.content contains the response data
            # This might need refinement based on the actual structure of CallToolResult
            # For simplicity, assuming tool_result.content is a list of fast-agent content types
            # and we need to represent this as a tool response in google.genai.
            # The 'response' in types.Part.from_function_response is a dict.
            # We need to structure the tool result content into a dictionary.

            # A simple approach: convert the tool result content to a string representation
            # A more robust approach: structure the tool result into a specific dictionary format
            # that the model can understand. This depends on the expected tool response format.

            # Structure the tool result content into a dictionary
            tool_response_data: Dict[str, Any] = {}
            for content_part in tool_result.content:
                if isinstance(content_part, TextContent):
                    if "text" not in tool_response_data:
                        tool_response_data["text"] = []
                    tool_response_data["text"].append(content_part.text)
                elif isinstance(content_part, EmbeddedResource):
                    if "resource" not in tool_response_data:
                        tool_response_data["resource"] = []
                    # Represent resource as a dictionary with uri and mimeType
                    resource_data = {}
                    if hasattr(content_part.resource, "uri") and content_part.resource.uri:
                        resource_data["uri"] = content_part.resource.uri
                    if (
                        hasattr(content_part.resource, "mimeType")
                        and content_part.resource.mimeType
                    ):
                        resource_data["mimeType"] = content_part.resource.mimeType
                    # Add other relevant resource attributes if needed
                    tool_response_data["resource"].append(resource_data)
                # Add handling for other content types in CallToolResult if necessary

            tool_response_part = types.Part.from_function_response(
                name=tool_name, response=tool_response_data
            )
            google_tool_response_contents.append(
                types.Content(role="tool", parts=[tool_response_part])
            )

        return google_tool_response_contents

    def convert_request_params_to_google_config(
        self, request_params: RequestParams
    ) -> types.GenerateContentConfig:
        """
        Converts fast-agent RequestParams to google.genai types.GenerateContentConfig.
        """
        config_args: Dict[str, Any] = {}

        # Map common parameters
        if request_params.temperature is not None:
            config_args["temperature"] = request_params.temperature
        if request_params.maxTokens is not None:
            # google.genai uses max_output_tokens
            config_args["max_output_tokens"] = request_params.maxTokens
        if hasattr(request_params, "topK") and request_params.topK is not None:
            config_args["top_k"] = request_params.topK
        if hasattr(request_params, "topP") and request_params.topP is not None:
            config_args["top_p"] = request_params.topP
        if hasattr(request_params, "stopSequences") and request_params.stopSequences is not None:
            config_args["stop_sequences"] = request_params.stopSequences
        if (
            hasattr(request_params, "presencePenalty")
            and request_params.presencePenalty is not None
        ):
            config_args["presence_penalty"] = request_params.presencePenalty
        if (
            hasattr(request_params, "frequencyPenalty")
            and request_params.frequencyPenalty is not None
        ):
            config_args["frequency_penalty"] = request_params.frequencyPenalty

        # System instruction
        if request_params.systemPrompt is not None:
            config_args["system_instruction"] = request_params.systemPrompt

        # Tool configuration will be handled in _google_completion based on available tools and parallel_tool_calls

        # Safety settings - assuming safety settings are part of request_params or global config
        # If they are in request_params, they need to be mapped to types.SafetySetting
        # if hasattr(request_params, 'safety_settings') and request_params.safety_settings:
        #     config_args['safety_settings'] = self.convert_to_google_safety_settings(request_params.safety_settings)

        # Create the GenerateContentConfig object
        return types.GenerateContentConfig(**config_args)

    def convert_from_google_content_list(
        self, contents: List[types.Content]
    ) -> List[PromptMessageMultipart]:
        """
        Converts a list of google.genai types.Content to a list of fast-agent PromptMessageMultipart.
        """
        return [self._convert_from_google_content(content) for content in contents]

    def _convert_from_google_content(self, content: types.Content) -> PromptMessageMultipart:
        """
        Converts a single google.genai types.Content to a fast-agent PromptMessageMultipart.
        """
        # If the content is a model's function call, create an assistant message with empty content
        # as tool call requests are handled separately for execution and not stored in history content.
        if content.role == "model" and any(part.function_call for part in content.parts):
            # Map google.genai 'model' role to fast-agent 'assistant' role
            return PromptMessageMultipart(role="assistant", content=[])

        fast_agent_parts: List[
            TextContent | ImageContent | EmbeddedResource | CallToolRequestParams
        ] = []

        for part in content.parts:
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_response:
                # Convert function response to TextContent for history
                # Assuming function_response.response is a dict, convert it to a string
                response_text = str(part.function_response.response)
                fast_agent_parts.append(TextContent(type="text", text=response_text))
            elif part.file_data:
                # Handle file_data (e.g., images from GCS URIs)
                # This requires fetching the content from the URI.
                # For now, we'll represent this as an EmbeddedResource with the URI.
                # Actual content fetching might happen elsewhere or be a future enhancement.
                fast_agent_parts.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextContent(  # Using TextContent for simplicity, might need a dedicated FileContent type
                            uri=part.file_data.file_uri,
                            mimeType=part.file_data.mime_type,
                            text=f"[Resource: {part.file_data.file_uri}, MIME: {part.file_data.mime_type}]",  # Placeholder text
                        ),
                    )
                )
            # Add handling for other part types if needed

        # Map google.genai roles to fast-agent roles for PromptMessageMultipart history
        # google.genai: 'user', 'model', 'tool'
        # fast-agent: 'user', 'assistant', 'tool', 'system'
        # 'system' role is not expected in model responses, handled in config.
        if content.role == "user":
            fast_agent_role = "user"
        else:
            # Map 'model' and 'tool' roles to 'assistant' for history
            fast_agent_role = "assistant"

        return PromptMessageMultipart(role=fast_agent_role, content=fast_agent_parts)

    # Helper method if safety settings need conversion
    # def convert_to_google_safety_settings(self, safety_settings_data) -> List[types.SafetySetting]:
    #      # Conversion logic from fast-agent safety settings format to google.genai types.SafetySetting
    #      pass
