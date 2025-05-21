#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from datetime import datetime


def analyze_runlog(folder_name, file_path):
    """
    Analyze a single runlog file and extract the required metrics.
    
    Args:
        folder_name: Name of the parent folder
        file_path: Path to the runlog file
        
    Returns:
        Dictionary containing the extracted metrics
    """
    try:
        # Extract run number from filename (e.g., runlog_5.jsonl -> 5)
        run_number = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            print(f"Warning: Empty file {file_path}", file=sys.stderr)
            return None
            
        # Parse each line as JSON
        parsed_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    parsed_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}: {line}", file=sys.stderr)
        
        if not parsed_lines:
            print(f"Warning: No valid JSON lines in {file_path}", file=sys.stderr)
            return None
            
        # Extract timestamps
        first_timestamp = datetime.fromisoformat(parsed_lines[0]["timestamp"])
        last_timestamp = datetime.fromisoformat(parsed_lines[-1]["timestamp"])
        elapsed_seconds = (last_timestamp - first_timestamp).total_seconds()
        
        # Extract token counts
        num_turns = len(parsed_lines)
        
        # Get token counts from message field
        def extract_token_counts(message):
            prompt_tokens = 0
            completion_tokens = 0
            
            if "Prompt Tokens:" in message and "Completion Tokens" in message:
                # Extract token counts using string parsing
                prompt_part = message.split("Prompt Tokens:")[1].split("Completion Tokens")[0].strip()
                completion_part = message.split("Completion Tokens :")[1].strip()
                
                try:
                    prompt_tokens = int(prompt_part)
                    completion_tokens = int(completion_part)
                except ValueError:
                    print(f"Warning: Could not parse token counts from message: {message}", file=sys.stderr)
            
            return prompt_tokens, completion_tokens
        
        # Extract token counts for all turns
        token_counts = [extract_token_counts(line["message"]) for line in parsed_lines]
        
        starting_prompt_tokens = token_counts[0][0] if token_counts else 0
        final_prompt_tokens = token_counts[-1][0] if token_counts else 0
        
        # Calculate cumulative tokens (sum of all prompt and completion tokens)
        cumulative_tokens = sum(pt + ct for pt, ct in token_counts)
        
        return {
            "folder": folder_name,
            "run_number": run_number,
            "elapsed_seconds": elapsed_seconds,
            "num_turns": num_turns,
            "starting_prompt_tokens": starting_prompt_tokens,
            "final_prompt_tokens": final_prompt_tokens,
            "cumulative_tokens": cumulative_tokens,
            "first_timestamp": first_timestamp.isoformat(),
            "last_timestamp": last_timestamp.isoformat(),
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
        return None


def process_folders(folder_names, output_file="runlog_summary.csv"):
    """
    Process all runlog files in the specified folders and generate a summary.
    
    Args:
        folder_names: List of folder names to process
        output_file: Path to the output CSV file
    """
    results = []
    
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            print(f"Warning: Folder {folder_name} does not exist", file=sys.stderr)
            continue
            
        # Find all runlog files in the folder
        runlog_files = [f for f in os.listdir(folder_name) if f.startswith("runlog_") and f.endswith(".jsonl")]
        
        for file_name in runlog_files:
            file_path = os.path.join(folder_name, file_name)
            result = analyze_runlog(folder_name, file_path)
            if result:
                results.append(result)
    
    # Sort results by folder name and run number
    results.sort(key=lambda x: (x["folder"], x["run_number"]))
    
    # Write results to CSV
    if results:
        fieldnames = results[0].keys()
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary written to {output_file}")
    else:
        print("No valid results found")


def main():
    parser = argparse.ArgumentParser(description="Analyze runlog files and generate a summary")
    parser.add_argument("folders", nargs='+', help="Folders containing runlog files")
    parser.add_argument("--output", "-o", default="runlog_summary.csv", 
                      help="Output CSV file (default: runlog_summary.csv)")
    
    args = parser.parse_args()
    process_folders(args.folders, args.output)


if __name__ == "__main__":
    main()