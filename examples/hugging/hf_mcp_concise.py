"""
WARNING: This is an experimental implementation. Expect rough edges while using it.
-------------------------------------------------
Defines a FastMCP server that exposes the Hugging Face Hub API as a set of tools.
In practice, all public methods from `HfApi` are exposed as tools, except for the ones dealing with files:
- `create_commit`
- `hf_hub_download`
- `preupload_lfs_files`
- `snapshot_download`
- `upload_file`
- `upload_folder`
- `upload_large_folder`
In addition, a `read_modelcard` tool is added to read the model card of a model on the Hugging Face Hub. Model card is
downloaded on the fly but not cached locally. If file is too large (> 1MB), it will raise an error.

You can specify which tools to include by setting the INCLUDED_TOOLS list. If empty, all tools will be included.
## How to use?
Use the MCP client of your choice to connect to the server.
You must pass the `HF_TOKEN` environment variable to the server.
Here is an example using `Agent` from `@huggingface/mcp-client` package:
```ts
const agent = new Agent({
        provider: "nebius",
        model: "Qwen/Qwen2.5-72B-Instruct",
        apiKey: process.env.HF_TOKEN,
        servers: [
                {
                        command: "python",
                        args: ["hf_mcp.py"],
                        env: {
                                HF_TOKEN: process.env.HF_TOKEN ?? "",
                        },
                },
        ],
});
```
## How it works?
Methods from `HfApi` are registered as tools in the FastMCP server. The `ctx` parameter is added to the method
signature to access the request context. The `token` parameter is removed from the methods signature and docstrings as
authentication is handled once in the context.
"""

import functools
import inspect
import re
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import requests
from huggingface_hub import HfApi, constants
from huggingface_hub.hf_api import *  # noqa: F403 # needed for tools parameter resolution
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

# Specify which tools to include. Leave empty to include all available tools.
# INCLUDED_TOOLS = ["whoami", "list_spaces", "get_space_runtime", "space_info"]
INCLUDED_TOOLS = ["whoami", "list_spaces", "get_space_runtime"]
# INCLUDED_TOOLS = []

REMOVE_TOKEN_RE = re.compile(
    r"""
    \n\s{12}token\s\(
    .*?
    (\n\s{12}[a-z])
    """,
    flags=re.VERBOSE | re.DOTALL | re.IGNORECASE | re.MULTILINE,
)

SKIPPED_METHODS = [
    "create_commit",
    "hf_hub_download",
    "preupload_lfs_files",
    "run_as_future",
    "snapshot_download",
    "upload_file",
    "upload_folder",
    "upload_large_folder",
    "list_models",
]


# special params: repo_type, revision, token
@dataclass
class AppContext:
    api: HfApi


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    yield AppContext(api=HfApi(library_name="huggingface-hub-mcp", library_version="0.0.1"))


mcp = FastMCP("Hugging Face Hub", lifespan=app_lifespan)


def register_hf_tool(api_name: str) -> None:
    api_method = getattr(HfApi, api_name)
    sig = inspect.signature(api_method)
    params = list(sig.parameters.values())

    # Remove `self` from the original method signature
    if params[0].name == "self":
        params = params[1:]

    # Tweak input parameters
    new_params = (
        # Add the `ctx` parameter
        [
            inspect.Parameter(
                "ctx",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Context[ServerSession, AppContext],
            )
        ]
        # Remote "token" parameter (handled in context)
        + [param for param in params if param.name != "token"]
    )

    new_sig = sig.replace(parameters=new_params)

    @functools.wraps(api_method)
    def wrapper(*args, **kwargs):
        bound_args = new_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        ctx = bound_args.arguments.pop("ctx")
        api = ctx.request_context.lifespan_context.api
        output = getattr(api, api_name)(**bound_args.arguments)
        # if output is a generator, convert it to a list
        if isinstance(output, typing.Generator):
            output = list(output)
            print(f"Found generator of length {len(output)}")
        return str(output)

    wrapper.__signature__ = new_sig

    new_doc = api_method.__doc__
    new_doc = REMOVE_TOKEN_RE.sub("\1", new_doc)
    wrapper.__doc__ = new_doc

    mcp.add_tool(wrapper)


@mcp.tool()
def read_modelcard(ctx: Context[ServerSession, AppContext], repo_id: str) -> str:
    """Read the model card of a Model on the Hugging Face Hub.
    If file is too large (> 1MB), it will raise an error.
    Args:
        repo_id: The ID of the repository.
    """
    # Download the repo card
    api = ctx.request_context.lifespan_context.api
    headers = api._build_hf_headers()

    # Build file URL
    url = constants.HUGGINGFACE_CO_URL_TEMPLATE.format(
        repo_id=repo_id, revision="main", filename="README.md"
    )

    # Check size
    response = requests.head(url, headers=headers)
    response.raise_for_status()
    size = int(response.headers["content-length"])
    if size > 1_000_000:
        raise ValueError(
            f"Model card for repo {repo_id} is too large to be read as text: {size} bytes"
        )

    # Download the file
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.content.decode("utf-8", errors="ignore")
    return content


# List methods from HfApi
methods = [
    name for name in dir(HfApi) if callable(getattr(HfApi, name)) and not name.startswith("_")
]


# Register tools
for name in sorted(dir(HfApi)):
    if name.startswith("_"):
        continue
    if name in SKIPPED_METHODS:
        continue
    # Skip if not in INCLUDED_TOOLS (if INCLUDED_TOOLS is not empty)
    if INCLUDED_TOOLS and name not in INCLUDED_TOOLS:
        continue

    method = getattr(HfApi, name)
    if inspect.iscoroutinefunction(method):
        continue
    if not callable(method):
        continue
    print(f"Registering {name}...")
    register_hf_tool(name)

if __name__ == "__main__":
    # Run the server
    mcp.run()
