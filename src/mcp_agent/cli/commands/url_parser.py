"""
URL parsing utility for the fast-agent CLI.
Provides functions to parse URLs and determine MCP server configurations.
"""

from typing import Dict, List, Literal, Tuple
from urllib.parse import urlparse
import re
import hashlib


def parse_server_url(
    url: str,
) -> Tuple[str, Literal["http", "sse"], str]:
    """
    Parse a server URL and determine the transport type and server name.
    
    Args:
        url: The URL to parse
        
    Returns:
        Tuple containing:
        - server_name: A generated name for the server
        - transport_type: Either "http" or "sse" based on URL
        - url: The parsed and validated URL
        
    Raises:
        ValueError: If the URL is invalid or unsupported
    """
    # Basic URL validation
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Ensure scheme is present and is either http or https
    if not parsed_url.scheme or parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"URL must have http or https scheme: {url}")
    
    # Ensure netloc (hostname) is present
    if not parsed_url.netloc:
        raise ValueError(f"URL must include a hostname: {url}")
    
    # Determine transport type based on URL path
    transport_type: Literal["http", "sse"] = "http"
    if parsed_url.path.endswith("/sse"):
        transport_type = "sse"
    elif not parsed_url.path.endswith("/mcp"):
        # If path doesn't end with /mcp or /sse, append /mcp
        url = url if url.endswith("/") else f"{url}/"
        url = f"{url}mcp"
    
    # Generate a server name based on hostname and port
    server_name = generate_server_name(url)
    
    return server_name, transport_type, url


def generate_server_name(url: str) -> str:
    """
    Generate a unique and readable server name from a URL.
    
    Args:
        url: The URL to generate a name for
        
    Returns:
        A server name string
    """
    parsed_url = urlparse(url)
    
    # Extract hostname and port
    hostname = parsed_url.netloc.split(":")[0]
    
    # Clean the hostname for use in a server name
    # Replace non-alphanumeric characters with underscores
    clean_hostname = re.sub(r'[^a-zA-Z0-9]', '_', hostname)
    
    # If it's localhost or an IP, add a more unique identifier
    if clean_hostname in ("localhost", "127_0_0_1") or re.match(r'^(\d+_){3}\d+$', clean_hostname):
        # Use the path as part of the name for uniqueness
        path = parsed_url.path.strip("/")
        path = re.sub(r'[^a-zA-Z0-9]', '_', path)
        
        # Include port if specified
        port = ""
        if ":" in parsed_url.netloc:
            port = f"_{parsed_url.netloc.split(':')[1]}"
        
        if path:
            return f"{clean_hostname}{port}_{path[:20]}"  # Limit path length
        else:
            # Use a hash if no path for uniqueness
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            return f"{clean_hostname}{port}_{url_hash}"
    
    return clean_hostname


def parse_server_urls(urls_param: str) -> List[Tuple[str, Literal["http", "sse"], str]]:
    """
    Parse a comma-separated list of URLs into server configurations.
    
    Args:
        urls_param: Comma-separated list of URLs
        
    Returns:
        List of tuples containing (server_name, transport_type, url)
        
    Raises:
        ValueError: If any URL is invalid
    """
    if not urls_param:
        return []
    
    # Split by comma and strip whitespace
    url_list = [url.strip() for url in urls_param.split(",")]
    
    # Parse each URL
    result = []
    for url in url_list:
        server_name, transport_type, parsed_url = parse_server_url(url)
        result.append((server_name, transport_type, parsed_url))
    
    return result


def generate_server_configs(
    parsed_urls: List[Tuple[str, Literal["http", "sse"], str]]
) -> Dict[str, Dict[str, str]]:
    """
    Generate server configurations from parsed URLs.
    
    Args:
        parsed_urls: List of tuples containing (server_name, transport_type, url)
        
    Returns:
        Dictionary of server configurations
    """
    server_configs = {}
    
    for server_name, transport_type, url in parsed_urls:
        server_configs[server_name] = {
            "transport": transport_type,
            "url": url,
        }
    
    return server_configs