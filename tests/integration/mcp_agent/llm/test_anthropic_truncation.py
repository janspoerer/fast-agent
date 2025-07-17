import pytest
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams


@pytest.mark.asyncio
async def test_simple_debug_find_issue():
    """
    Simple test to trigger the bug and examine the exact line where it happens.
    """
    print("=== Simple debug to find the issue ===")
    
    fast = FastAgent("simple_debug_test", parse_cli_args=False, quiet=True)

    params_dict = {
        "max_context_tokens": 50,  # Low limit to trigger the bug
        "truncation_strategy": "summarize"
    }

    request_params = RequestParams(**params_dict)

    @fast.agent(
        name="simple_debug_agent",
        model="claude-3-haiku-20240307",
        request_params=request_params
    )
    async def simple_debug_agent(agent):
        """
        Simple agent to trigger the bug.
        """
        return "Simple debug response"

    async with fast.run() as app:
        agent = app.simple_debug_agent
        
        # Send a message that will definitely trigger truncation
        long_message = f"{'This message is designed to trigger the truncation bug. ' * 10}"
        print(f"Sending message that should trigger truncation bug...")
        
        try:
            response = await agent.send(long_message)
            print(f"Unexpected success: {response}")
        except AttributeError as e:
            if "'dict' object has no attribute 'first_text'" in str(e):
                print("✓ Successfully reproduced the bug!")
                print(f"Error: {e}")
                
                # Now let's examine the call stack
                import traceback
                print("\n=== CALL STACK ANALYSIS ===")
                tb = traceback.format_exc()
                print(tb)
                
                # Look for the specific line that's causing trouble
                lines = tb.split('\n')
                for i, line in enumerate(lines):
                    if 'self.history.extend(converted' in line:
                        print(f"\n=== FOUND THE PROBLEM LINE ===")
                        print(f"Line {i}: {line}")
                        print("This is where 'converted' (a list of dicts) gets stored!")
                        break
                    elif 'augmented_llm_anthropic.py' in line and 'extend' in line:
                        print(f"\n=== FOUND RELATED LINE ===")
                        print(f"Line {i}: {line}")
                
                print("\n=== CONCLUSION ===")
                print("The bug is in the Anthropic provider where it stores")
                print("'converted' (dict format) instead of original PromptMessageMultipart objects")
                
            else:
                print(f"Different error: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"Other error: {e}")
            traceback.print_exc()


@pytest.mark.asyncio
async def test_show_exact_line_numbers():
    """
    Test to show the exact line numbers and file content where the issue occurs.
    """
    print("=== Finding exact line numbers ===")
    
    import inspect
    import os
    
    # Try to find the augmented_llm_anthropic file
    try:
        from mcp_agent.llm.providers import augmented_llm_anthropic
        file_path = inspect.getfile(augmented_llm_anthropic)
        print(f"Found file: {file_path}")
        
        # Read the file and look for the problematic line
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Look for lines around 521 (mentioned in the error)
            target_line = 521
            start_line = max(0, target_line - 10)
            end_line = min(len(lines), target_line + 10)
            
            print(f"\n=== LINES {start_line}-{end_line} FROM {file_path} ===")
            for i in range(start_line, end_line):
                if i < len(lines):
                    line_num = i + 1
                    line_content = lines[i].rstrip()
                    marker = " <-- PROBLEM LINE" if 'extend(converted' in line_content else ""
                    print(f"{line_num:3d}: {line_content}{marker}")
        
    except ImportError as e:
        print(f"Could not import module: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")


@pytest.mark.asyncio
async def test_reproduce_and_analyze():
    """
    Reproduce the bug and provide analysis of what needs to be fixed.
    """
    print("=== Reproduce and analyze ===")
    
    fast = FastAgent("analyze_test", parse_cli_args=False, quiet=True)

    params_dict = {
        "max_context_tokens": 100,
        "truncation_strategy": "summarize"
    }

    request_params = RequestParams(**params_dict)

    @fast.agent(
        name="analyze_agent",
        model="claude-3-haiku-20240307",
        request_params=request_params
    )
    async def analyze_agent(agent):
        return "Analysis response"

    async with fast.run() as app:
        agent = app.analyze_agent
        
        long_message = f"{'Long message to trigger truncation. ' * 20}"
        
        try:
            response = await agent.send(long_message)
            print("No error occurred - truncation might not have been triggered")
        except Exception as e:
            print(f"Error occurred: {e}")
            print("\n=== ANALYSIS ===")
            print("1. The error happens in context_truncation.py line 70")
            print("2. It tries to call message.first_text() on a dict")
            print("3. This means memory.get() returns dicts instead of PromptMessageMultipart")
            print("4. The root cause is in augmented_llm_anthropic.py around line 521")
            print("5. The 'converted' variable contains dicts (for API format)")
            print("6. But these dicts get stored in memory instead of original objects")
            print("\n=== SOLUTION ===")
            print("The Anthropic provider should store the original PromptMessageMultipart")
            print("objects in memory, not the converted dict format used for API calls.")
            
            return  # Don't re-raise, we got the info we needed