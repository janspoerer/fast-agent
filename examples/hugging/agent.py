import asyncio

from opentelemetry import trace

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams

# Create the application
fast = FastAgent("fast-agent example")
model = "sonnet"


# Define the agent
@fast.agent(
    name="standard",
    instruction="You are a helpful AI Agent",
    servers=["hf_mcp_complete"],
    request_params=RequestParams(maxTokens=8192),
    model=model,
)
@fast.agent(
    name="concise",
    instruction="You are a helpful AI Agent",
    servers=["hf_search_tools"],
    request_params=RequestParams(maxTokens=8192),
    model=model,
)
@fast.agent(
    name="interfere",
    instruction="You are a helpful AI Agent",
    servers=["hf_search_tools", "filesystem", "github"],
    request_params=RequestParams(maxTokens=8192),
    model=model,
)
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        tracer = trace.get_tracer(__name__)
        #        await agent.interactive()
        #    with tracer.start_as_current_span(
        #     f"Standard {agent.standard._llm.default_request_params.model}"
        # ):
        #     await agent.standard.send("who am i on hugging face")
        #     await agent.standard.send("what spaces do i have?")
        #     await agent.standard.send("how long has my hugging face space jaguar been running?")

        # with tracer.start_as_current_span(
        #     f"Concise {agent.concise._llm.default_request_params.model}"
        # ):
        #     await agent.concise.send("find me papers on the 'kazakh language'")
        #     await agent.concise.send("are there any related models?")
        #     await agent.concise.send("are there any spaces?")

        with tracer.start_as_current_span(
            f"Concise and fs/github {agent.concise._llm.default_request_params.model}"
        ):
            await agent.interfere.send("find me papers on the 'kazakh language'")
            await agent.interfere.send("are there any related models?")
            await agent.interfere.send("are there any spaces?")


if __name__ == "__main__":
    asyncio.run(main())
