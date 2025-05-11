"""Run an interactive agent directly from the command line."""

import asyncio
import sys
from typing import Dict, List, Optional

import typer

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.cli.commands.url_parser import parse_server_urls, generate_server_configs

app = typer.Typer(
    help="Run an interactive agent directly from the command line without creating an agent.py file"
)

async def _run_agent(
    name: str = "FastAgent CLI",
    instruction: str = "You are a helpful AI Agent.",
    config_path: Optional[str] = None,
    server_list: Optional[List[str]] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    prompt_file: Optional[str] = None,
    url_servers: Optional[Dict[str, Dict[str, str]]] = None
) -> None:
    """Async implementation to run an interactive agent."""
    from pathlib import Path
    import os

    from mcp_agent.mcp.prompts.prompt_load import load_prompt_multipart
    from mcp_agent.config import MCPServerSettings, MCPSettings, Settings

    # Create the FastAgent instance
    fast_kwargs = {
        "name": name,
        "config_path": config_path,
        "ignore_unknown_args": True,
        "parse_cli_args": False,  # Don't parse CLI args, we're handling it ourselves
    }
    
    fast = FastAgent(**fast_kwargs)
    
    # Add URL-based servers to the context configuration
    if url_servers:
        # Initialize the app to ensure context is ready
        await fast.app.initialize()
        
        # Initialize mcp settings if needed
        if not hasattr(fast.app.context.config, "mcp"):
            fast.app.context.config.mcp = MCPSettings()
            
        # Initialize servers dictionary if needed
        if not hasattr(fast.app.context.config.mcp, "servers") or fast.app.context.config.mcp.servers is None:
            fast.app.context.config.mcp.servers = {}
            
        # Add each URL server to the config
        for server_name, server_config in url_servers.items():
            fast.app.context.config.mcp.servers[server_name] = MCPServerSettings(
                transport=server_config["transport"],
                url=server_config["url"]
            )

    # Define the agent with specified parameters
    agent_kwargs = {"instruction": instruction}
    if server_list:
        agent_kwargs["servers"] = server_list
    if model:
        agent_kwargs["model"] = model
    
    # Handle prompt file and message options
    if message or prompt_file:
        @fast.agent(**agent_kwargs)
        async def cli_agent():
            async with fast.run() as agent:
                if message:
                    response = await agent.send(message)
                    # Print the response and exit
                    print(response)
                elif prompt_file:
                    prompt = load_prompt_multipart(Path(prompt_file))
                    response = await agent.generate(prompt)
                    # Print the response text and exit
                    print(response.last_text())
    else:
        # Standard interactive mode
        @fast.agent(**agent_kwargs)
        async def cli_agent():
            async with fast.run() as agent:
                await agent.interactive()

    # Run the agent
    await cli_agent()

def run_async_agent(
    name: str,
    instruction: str,
    config_path: Optional[str] = None,
    servers: Optional[str] = None,
    urls: Optional[str] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    prompt_file: Optional[str] = None
):
    """Run the async agent function with proper loop handling."""
    server_list = servers.split(',') if servers else None

    # Parse URLs and generate server configurations if provided
    url_servers = None
    if urls:
        try:
            parsed_urls = parse_server_urls(urls)
            url_servers = generate_server_configs(parsed_urls)
            # If we have servers from URLs, add their names to the server_list
            if url_servers and not server_list:
                server_list = list(url_servers.keys())
            elif url_servers and server_list:
                # Merge both lists
                server_list.extend(list(url_servers.keys()))
        except ValueError as e:
            print(f"Error parsing URLs: {e}")
            return

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside a running event loop, so we can't use asyncio.run
            # Instead, create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No event loop exists, so we'll create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_run_agent(
            name=name,
            instruction=instruction,
            config_path=config_path,
            server_list=server_list,
            model=model,
            message=message,
            prompt_file=prompt_file,
            url_servers=url_servers
        ))
    finally:
        try:
            # Clean up the loop
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            
            # Run the event loop until all tasks are done
            if sys.version_info >= (3, 7):
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception:
            pass

@app.callback(invoke_without_command=True)
def go(
    ctx: typer.Context,
    name: str = typer.Option("FastAgent CLI", "--name", help="Name for the agent"),
    instruction: str = typer.Option(
        "You are a helpful AI Agent.", "--instruction", "-i", help="Instruction for the agent"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config-path", "-c", help="Path to config file"
    ),
    servers: Optional[str] = typer.Option(
        None, "--servers", help="Comma-separated list of server names to enable from config"
    ),
    urls: Optional[str] = typer.Option(
        None, "--url", help="Comma-separated list of HTTP/SSE URLs to connect to"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override the default model (e.g., haiku, sonnet, gpt-4)"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Message to send to the agent (skips interactive mode)"
    ),
    prompt_file: Optional[str] = typer.Option(
        None, "--prompt-file", "-p", help="Path to a prompt file to use (either text or JSON)"
    ),
) -> None:
    """
    Run an interactive agent directly from the command line.

    Examples:
        fast-agent go --model=haiku --instruction="You are a coding assistant" --servers=fetch,filesystem
        fast-agent go --message="What is the weather today?" --model=haiku
        fast-agent go --prompt-file=my-prompt.txt --model=haiku
        fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

    This will start an interactive session with the agent, using the specified model
    and instruction. It will use the default configuration from fastagent.config.yaml
    unless --config-path is specified.

    Common options:
        --model               Override the default model (e.g., --model=haiku)
        --quiet               Disable progress display and logging
        --servers             Comma-separated list of server names to enable from config
        --url                 Comma-separated list of HTTP/SSE URLs to connect to
        --message, -m         Send a single message and exit
        --prompt-file, -p     Use a prompt file instead of interactive mode
    """
    run_async_agent(
        name=name,
        instruction=instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        model=model,
        message=message,
        prompt_file=prompt_file
    )