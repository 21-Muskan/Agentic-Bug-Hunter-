"""
Bug Hunter Agent Client (Agentic RAG Edition)
----------------------------------------------
MCP client that connects to the MCP server, uses `search_documents`
to fetch relevant documentation (RAG), runs static analysis (CppCheck),
then calls the Hugging Face Inference API to analyze code for bugs.
"""

import os
import ast
import csv
import json
import asyncio
import argparse
import httpx
import pandas as pd
from dotenv import load_dotenv
from fastmcp import Client
from utils.code_analyzer import (
    add_line_numbers,
    build_analysis_prompt,
    parse_llm_response,
    format_rag_docs,
)
from utils.cpp_checker import check_code_snippet

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8003/sse")

# HF Router API endpoint (the new unified endpoint)
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"


class BugHunterAgent:
    """Agentic AI client that orchestrates MCP tools + LLM + Static Analysis for bug detection."""

    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url

    async def search_docs(self, mcp_client: Client, query: str) -> list:
        """Call MCP search_documents to retrieve relevant documentation."""
        try:
            result = await mcp_client.call_tool("search_documents", {"query": query})
            content_list = result.content if hasattr(result, 'content') else []

            if content_list and len(content_list) > 0:
                raw_text = content_list[0].text if hasattr(content_list[0], 'text') else str(content_list[0])
                try:
                    docs = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
                    if isinstance(docs, list):
                        return docs
                except (json.JSONDecodeError, ValueError):
                    try:
                        docs = ast.literal_eval(raw_text)
                        if isinstance(docs, list):
                            return docs
                    except:
                        pass
            return []
        except Exception as e:
            print(f"    [WARN] search_documents failed: {e}")
            return []

    def call_llm(self, prompt: str) -> str:
        """Call Hugging Face Inference API directly via httpx."""
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": HF_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(HF_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # OpenAI-compatible chat response format
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]

            # Fallback
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", str(data[0]))

            raise RuntimeError(f"Unexpected HF API response format: {json.dumps(data)[:200]}")

    async def analyze_entry(self, mcp_client: Client, entry_id: str, code: str, context: str) -> dict:
        """Analyze a single code entry using RAG + Static Analysis + LLM."""
        try:
            # Step 1: RAG — search for relevant documentation
            print(f"    -> Searching knowledge base for: '{context[:60]}...'")
            rag_results = await self.search_docs(mcp_client, context)
            rag_text = format_rag_docs(rag_results, max_docs=5)
            if rag_results:
                print(f"    -> Retrieved {len(rag_results)} docs (using top 5)")
            else:
                print(f"    -> No docs retrieved, proceeding with context only")

            # Step 2: Run Static Analysis (CppCheck)
            print(f"    -> Running CppCheck...")
            static_errors = check_code_snippet(code)
            if static_errors:
                print(f"    -> CppCheck detected issues:\n{static_errors}")
            else:
                print(f"    -> CppCheck passed (no issues found)")

            # Step 3: Add line numbers
            numbered_code = add_line_numbers(code)

            # Step 4: Build enriched prompt
            prompt = build_analysis_prompt(numbered_code, context, rag_text, static_errors)

            # Step 5: Call LLM
            print(f"    -> Calling LLM ({HF_MODEL_ID})...")
            response_text = self.call_llm(prompt)

            # Step 6: Parse response
            result = parse_llm_response(response_text)

            bug_lines = result.get("bug_lines", [])
            explanations = result.get("explanations", [])
            corrected_code = result.get("corrected_code", "")

            return {
                "ID": entry_id,
                "Bug Line": ",".join(str(l) for l in bug_lines) if bug_lines else "",
                "Explanation": "; ".join(explanations) if explanations else "No bugs detected",
                "Corrected Code": corrected_code,
                "RAG Docs": rag_results  # Return for UI inspection
            }

        except Exception as e:
            print(f"    ERROR: {e}")
            return {
                "ID": entry_id,
                "Bug Line": "",
                "Explanation": f"Error: {str(e)}",
                "Corrected Code": "",
                "RAG Docs": []
            }

    async def analyze_single_snippet(self, code: str, context: str) -> dict:
        """Analyze a single snippet without ID (for UI)."""
        async with Client(self.server_url) as mcp_client:
            return await self.analyze_entry(mcp_client, "UI_Request", code, context)

    async def process_csv(self, input_file: str, output_file: str = "data/output.csv"):
        """Process the input CSV and write results to output CSV."""
        print(f"\n{'='*60}")
        print(f"  Bug Hunter Agent -- Agentic RAG + CppCheck Bug Detector")
        print(f"{'='*60}")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")
        print(f"  MCP:    {self.server_url}")
        print(f"  LLM:    {HF_MODEL_ID}")
        print(f"{'='*60}\n")

        # Read the input CSV
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} entries from {input_file}\n")

        results = []

        # Connect to MCP server
        async with Client(self.server_url) as mcp_client:
            print("Connected to MCP server!")

            # List available tools
            tools = await mcp_client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"Available MCP tools: {tool_names}\n")

            # Process each entry
            for index, row in df.iterrows():
                entry_id = str(row["ID"])
                context = str(row.get("Context", ""))
                code = str(row.get("Code", ""))

                print(f"[{index + 1}/{len(df)}] Analyzing ID={entry_id}...")

                result = await self.analyze_entry(mcp_client, entry_id, code, context)
                # Remove non-serializable objects for CSV
                csv_result = result.copy()
                csv_result.pop("RAG Docs", None) 
                results.append(csv_result)

                # Show brief result
                if result["Bug Line"]:
                    print(f"    >> Bugs at line(s): {result['Bug Line']}")
                    print(f"       {result['Explanation'][:120]}...")
                else:
                    print(f"    >> No bugs detected")
                print()

        # Write results to CSV
        output_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        # Ensure 'Corrected Code' is in output
        columns = ["ID", "Bug Line", "Explanation", "Corrected Code"]
        # Filter for existing columns only in case others were added
        cols_to_use = [c for c in columns if c in output_df.columns]
        output_df = output_df[cols_to_use]
        
        output_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

        # Summary
        print(f"\n{'='*60}")
        print(f"  Analysis Complete!")
        print(f"{'='*60}")
        bugs_found = sum(1 for r in results if r["Bug Line"])
        print(f"  Total entries:    {len(results)}")
        print(f"  Bugs detected in: {bugs_found} entries")
        print(f"  Results saved to: {output_file}")
        print(f"{'='*60}\n")

        return output_file


def main():
    parser = argparse.ArgumentParser(description="Bug Hunter Agentic AI Client")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="data/output.csv", help="Output CSV file path")
    parser.add_argument("--server", default=MCP_SERVER_URL, help="MCP server SSE URL")

    args = parser.parse_args()

    agent = BugHunterAgent(args.server)
    asyncio.run(agent.process_csv(args.input, args.output))


if __name__ == "__main__":
    main()
