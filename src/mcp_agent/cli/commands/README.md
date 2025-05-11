# Fast-Agent CLI Commands

This directory contains the command implementations for the fast-agent CLI.

## Go Command

The `go` command allows you to run an interactive agent directly from the command line without
creating a dedicated agent.py file.

### Usage

```bash
fast-agent go [OPTIONS]
```

### Options

- `--name TEXT`: Name for the agent (default: "FastAgent CLI")
- `--instruction`, `-i TEXT`: Instruction for the agent (default: "You are a helpful AI Agent.")
- `--config-path`, `-c TEXT`: Path to config file
- `--servers TEXT`: Comma-separated list of server names to enable from config
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to directly
- `--model TEXT`: Override the default model (e.g., haiku, sonnet, gpt-4)
- `--message`, `-m TEXT`: Message to send to the agent (skips interactive mode)
- `--prompt-file`, `-p TEXT`: Path to a prompt file to use (either text or JSON)
- `--quiet`: Disable progress display and logging

### Examples

```bash
# Basic usage with interactive mode
fast-agent go --model=haiku

# Specifying servers from configuration
fast-agent go --servers=fetch,filesystem --model=haiku

# Directly connecting to HTTP/SSE servers via URLs
fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

# Non-interactive mode with a single message
fast-agent go --message="What is the weather today?" --model=haiku

# Using a prompt file
fast-agent go --prompt-file=my-prompt.txt --model=haiku
```

### URL Connection Details

The `--url` parameter allows you to connect directly to HTTP or SSE servers using URLs.

- URLs must have http or https scheme
- The transport type is determined by the URL path:
  - URLs ending with `/sse` are treated as SSE transport
  - URLs ending with `/mcp` or automatically appended with `/mcp` are treated as HTTP transport
- Server names are generated automatically based on the hostname, port, and path
- The URL-based servers are added to the agent's configuration and enabled