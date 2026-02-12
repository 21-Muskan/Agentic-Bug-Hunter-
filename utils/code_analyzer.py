"""
Code Analyzer Utilities
-----------------------
Helper functions for preparing code for LLM analysis and parsing responses.
"""

import json
import re


def add_line_numbers(code: str) -> str:
    """Add line numbers to each line of the code snippet."""
    lines = code.split("\n")
    numbered = [f"{i + 1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def build_analysis_prompt(numbered_code: str, context: str, rag_docs: str = "", static_errors: str = "") -> str:
    """Build the LLM prompt for bug detection with few-shot examples and static analysis input."""
    rag_section = ""
    if rag_docs and rag_docs.strip():
        rag_section = f"""
**Relevant API Documentation (from knowledge base):**
{rag_docs}
"""

    static_section = ""
    if static_errors and static_errors.strip():
        static_section = f"""
**Static Analysis Findings (CppCheck):**
{static_errors}
(Note: Use these findings as strong evidence, but verify them against the context.)
"""

    return f"""You are an expert bug detector for RDI/SmartRDI embedded test code. Find ALL bugs and provide the CORRECTED code.

**Context:** {context}
{rag_section}
{static_section}
**Code:**
```
{numbered_code}
```

**Bug categories to check:**
- Wrong/misspelled function names (e.g., readHumanSeniority vs readHumSensor, iMeans vs iMeas)
- Wrong argument order (e.g., iClamp(high, low) should be iClamp(low, high))
- Values exceeding documented ranges.
- Wrong API calls (e.g., use execute() instead of burst()).
- Variable mismatches.
- Lifecycle errors (e.g., RDI_END before RDI_BEGIN).
- Pin name typos (e.g., "D0" vs "DO").

**FEW-SHOT EXAMPLES:**

Example 1:
Code: `2: rdi.pmux(4).module("02").readHumanSeniority().execute();`
Output: {{"bug_lines": [2], "explanations": ["readHumanSeniority -> readHumSensor"], "corrected_code": "rdi.pmux(4).module(\\"02\\").readHumSensor().execute();"}}

Example 2:
Code: `3: iClamp(50 mA, -50 mA);`
Output: {{"bug_lines": [3], "explanations": ["iClamp args swapped"], "corrected_code": "iClamp(-50 mA, 50 mA);"}}

**RULES:**
1. Explanations MUST be under 10 words.
2. Report the exact line number.
3. Provide the COMPLETE corrected line(s) of code in `corrected_code`. If multiple lines are wrong, fix all of them.
4. Respond with ONLY the JSON object.

{{"bug_lines": [line_numbers], "explanations": ["short explanation"], "corrected_code": "full corrected code snippet"}}"""


def parse_llm_response(response: str) -> dict:
    """Parse the LLM response into structured bug data."""
    default = {"bug_lines": [], "explanations": [], "corrected_code": ""}

    if not response or not response.strip():
        return default

    text = response.strip()

    # Try to extract JSON from markdown code fences if present
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Try to find a JSON object in the text
    json_obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_obj_match:
        text = json_obj_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _fallback_parse(response)

    bug_lines = data.get("bug_lines", [])
    explanations = data.get("explanations", [])
    corrected_code = data.get("corrected_code", "")

    # Ensure bug_lines are strings
    bug_lines = [str(line) for line in bug_lines]

    # Truncate long explanations
    explanations = [_truncate(e, max_words=15) for e in explanations]

    # Ensure lists are same length
    while len(explanations) < len(bug_lines):
        explanations.append("Bug detected")
    while len(bug_lines) < len(explanations):
        bug_lines.append("")

    return {"bug_lines": bug_lines, "explanations": explanations, "corrected_code": corrected_code}


def _truncate(text: str, max_words: int = 15) -> str:
    """Truncate explanation to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _fallback_parse(response: str) -> dict:
    """Fallback parser when JSON extraction fails."""
    bug_lines = []
    explanations = []
    corrected_code = ""

    line_pattern = re.compile(
        r"[Ll]ine\s+(\d+)\s*[:\-\u2013]\s*(.+?)(?:\n|$)", re.MULTILINE
    )
    for match in line_pattern.finditer(response):
        bug_lines.append(match.group(1))
        explanations.append(match.group(2).strip())
    
    # Try to find corrected code block
    code_match = re.search(r"Corrected Code:?\s*```(?:cpp|c)?\n?(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if code_match:
        corrected_code = code_match.group(1).strip()

    return {"bug_lines": bug_lines, "explanations": explanations, "corrected_code": corrected_code}


def format_rag_docs(rag_results: list, max_docs: int = 5) -> str:
    """Format RAG search results into a clean documentation string."""
    if not rag_results:
        return ""

    sorted_docs = sorted(rag_results, key=lambda x: x.get("score", 0), reverse=True)
    top_docs = sorted_docs[:max_docs]

    parts = []
    for i, doc in enumerate(top_docs, 1):
        text = doc.get("text", "").strip()
        score = doc.get("score", 0)
        if text:
            parts.append(f"[Doc {i} (relevance: {score:.3f})]:\n{text}")

    return "\n\n".join(parts)
