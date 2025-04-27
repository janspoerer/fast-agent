import asyncio
from typing import Annotated

from pydantic import BaseModel, Field

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("fast-agent example")


class FormattedResponse(BaseModel):
    thinking: Annotated[
        str, Field(description="Your reflection on the conversation that is not seen by the user.")
    ]
    message: str


# Define the agent
@fast.agent(
    name="chat",
    instruction="You are a helpful AI Agent",
    request_params=(RequestParams(maxTokens=8192)),
)
@fast.chain(
    name="chain",
    sequence=["chat", "route"],
)
@fast.agent(
    name="fetcher",
    instruction="""You are an agent, with a tool enabling you to fetch URLs.""",
    servers=["fetch"],
)
@fast.agent(
    name="general_assistant",
    instruction="""You are a knowledgeable assistant that provides clear,
    well-reasoned responses about general topics, concepts, and principles.""",
)
@fast.router(
    name="route",
    model="sonnet",
    agents=["general_assistant", "fetcher"],
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
