import pytest
from typing import List

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart, TextContent


def calculate_text_length(history: List[PromptMessageMultipart]) -> int:
    """Calculates total length of all TextContent in a list of messages."""
    length = 0
    for message in history:
        for content_piece in message.content:
            if isinstance(content_piece, TextContent) and content_piece.text:
                length += len(content_piece.text)
    return length


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_runtime_kwargs_combined_truncation_gemini(fast_agent):
    """
    Test that runtime kwargs for truncation parameters are correctly used.
    This test verifies both max_context_length_per_message and max_total_context_length
    are applied correctly when passed as keyword arguments to agent().
    """
    fast = fast_agent # Assign to 'fast' for consistency with other tests

    agent_name_or_model = "gemini25" # This model name will be used in the decorator

    @fast.agent(
        "truncation_agent", # Give your agent a unique name for this test
        instruction="You are a helpful AI Agent designed for testing context truncation.",
        model=agent_name_or_model,
    )
    async def truncation_agent_function():
        async with fast.run() as agent_session:
            # Access the specific agent instance by its name
            agent_instance = agent_session.truncation_agent

            # Explicitly set these config parameters to None on the agent's config
            # as intended by your original test to ensure runtime override is tested.
            agent_instance.config.max_context_length_per_message = None
            agent_instance.config.max_total_context_length = None

            # Create test messages designed to be affected by truncation
            long_text_1 = "This is a very long message that should be truncated to exactly fifteen characters when the max_context_length_per_message is set to 15."
            long_text_2 = "This is another long message that should also be truncated to 15 characters."
            short_reply = "Short reply."

            # Initialize LLM if needed (as per your original logic, using agent.send)
            # Access _llm through agent_instance
            if not hasattr(agent_instance, "_llm") or not agent_instance._llm:
                try:
                    # **IMPORTANT FIX HERE:** Call send on agent_instance, not agent_session
                    await agent_instance.send(Prompt.user("Initialize LLM"))
                except Exception:
                    pass
                if not hasattr(agent_instance, "_llm") or not agent_instance._llm:
                    pytest.fail("Agent LLM could not be initialized")

            # Manually set the history before the call that triggers truncation
            # Access _llm._message_history through agent_instance
            agent_instance._llm._message_history = [
                Prompt.user(long_text_1),
                Prompt.assistant(short_reply),
                Prompt.user(long_text_2)
            ]

            # Define runtime truncation parameters
            runtime_max_len_per_msg = 15
            runtime_max_total_len = 40

            # Call the agent with runtime kwargs
            trigger_message_text = "Trigger message"
            # **IMPORTANT FIX HERE:** Call send on agent_instance, not agent_session
            await agent_instance.send(
                Prompt.user(trigger_message_text),
                max_context_length_per_message=runtime_max_len_per_msg,
                max_total_context_length=runtime_max_total_len
            )

            # Find the trigger message to accurately slice history
            final_history = agent_instance._llm._message_history # Access through agent_instance
            trigger_message_index = -1
            for i, msg in enumerate(final_history):
                if msg.role == "user" and msg.content and isinstance(msg.content[0], TextContent) and trigger_message_text in msg.content[0].text:
                    trigger_message_index = i
                    break

            assert trigger_message_index != -1, "Trigger message not found in final history"

            # The messages before the trigger message are the ones that went through truncation
            truncated_history = final_history[:trigger_message_index]

            # Verify that the total length of truncated_history meets the max_total_context_length constraint
            total_length = calculate_text_length(truncated_history)
            assert total_length <= runtime_max_total_len, f"Total length {total_length} exceeds max {runtime_max_total_len}"

            # Verify message-level truncation
            for msg in truncated_history:
                for content in msg.content:
                    if isinstance(content, TextContent) and content.text:
                        if msg.role == "user":
                            assert len(content.text) <= runtime_max_len_per_msg, \
                                f"Message length {len(content.text)} exceeds per-message max {runtime_max_len_per_msg}"

                        if msg.role == "user" and long_text_2[:5] in content.text:
                            assert content.text == long_text_2[:runtime_max_len_per_msg], \
                                "Second user message not truncated correctly"

                        if msg.role == "assistant" and content.text == short_reply:
                            assert content.text == short_reply, "Short reply was incorrectly modified"

            # Test a case where total length causes message removal
            # Reset config on agent_instance
            agent_instance.config.max_context_length_per_message = None
            agent_instance.config.max_total_context_length = None

            # Reset the history on agent_instance
            agent_instance._llm._message_history = [
                Prompt.user(long_text_1),
                Prompt.assistant(short_reply),
                Prompt.user(long_text_2)
            ]

            # Set a very small total length limit that will force message removal
            small_total_limit = 20

            # Call the agent with the small total length limit
            second_trigger = "Second trigger"
            # **IMPORTANT FIX HERE:** Call send on agent_instance, not agent_session
            await agent_instance.send(
                Prompt.user(second_trigger),
                max_context_length_per_message=runtime_max_len_per_msg,
                max_total_context_length=small_total_limit
            )

            # Find the second trigger message
            final_history_2 = agent_instance._llm._message_history # Access through agent_instance
            trigger_2_index = -1
            for i, msg in enumerate(final_history_2):
                if msg.role == "user" and msg.content and isinstance(msg.content[0], TextContent) and second_trigger in msg.content[0].text:
                    trigger_2_index = i
                    break

            assert trigger_2_index != -1, "Second trigger message not found in final history"

            # Check that the total length of messages before the second trigger is within the small limit
            truncated_history_2 = final_history_2[:trigger_2_index]
            total_length_2 = calculate_text_length(truncated_history_2)
            assert total_length_2 <= small_total_limit, \
                f"Total length {total_length_2} exceeds small max {small_total_limit}"

            # The history should have fewer messages due to the small total limit
            assert len(truncated_history_2) < len(truncated_history), \
                "Expected fewer messages with smaller total length limit"

            # Reset for cleanup
            agent_instance.config.max_context_length_per_message = None
            agent_instance.config.max_total_context_length = None

    # Call the defined agent function to execute the test logic
    await truncation_agent_function()


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_agent_definition_truncation_parameters(fast_agent):
    fast = fast_agent

    # Define an agent with the truncation parameters directly in the decorator
    @fast.agent(
        "configurable_truncation_agent",
        instruction="Agent with static context truncation.",
        model="gemini25",
        max_total_context_length=50, # Example static values
        max_context_length_per_message=10, # Example static values
    )
    async def static_truncation_agent_function():
        async with fast.run() as agent_session:
            agent_instance = agent_session.configurable_truncation_agent # Access by its specific name

            # Assert that the agent's config reflects these values
            assert agent_instance.config.max_total_context_length == 50
            assert agent_instance.config.max_context_length_per_message == 10

            # (Optional) You could then proceed to send messages and verify truncation based on these *static* values.
            # E.g., send a very long message and assert its length in history.
            # This would be a complementary test to your existing one.

    await static_truncation_agent_function()