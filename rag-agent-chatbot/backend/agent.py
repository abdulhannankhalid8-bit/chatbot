"""
Agentic RAG: tools = search_documents (+ optional web_search).
Uses Groq native tool-calling API.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from groq import AsyncGroq

from backend.rag import format_context, search_documents

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_AGENT_STEPS = 5
ENABLE_WEB = os.getenv("ENABLE_WEB_SEARCH", "1").strip() not in {"0", "false", "False"}

SYSTEM_PROMPT = """You are a friendly, expressive assistant with tools. Talk like a helpful teammate — warm, clear, and a little human (light enthusiasm is good; no robotic stiff tone).

TOOLS (use them when needed — they work the same for every model):
1) search_documents — search the user's uploaded knowledge base (PDFs/TXT).
2) web_search — search the public internet (when available in the tool list).

WHEN TO USE TOOLS:
- Questions about uploaded files / company / private docs → search_documents first.
- General knowledge, definitions, current events, "how does X work", tutorials → web_search when available (or your knowledge if web is off).
- You may use both tools in one turn if useful.
- NEVER claim you searched the web or documents unless you actually called that tool.

CONVERSATION RULES (important):
- Short acknowledgements like "okay", "ok", "got it", "thanks", "cool", "alright", "yes", "hmm" → do NOT repeat your previous long answer. Reply briefly and naturally (e.g. "Got it! 🙂 What would you like next?" / "Sure thing — ask me anything else.").
- Greetings ("hi", "hello") → short friendly hello; offer to help.
- Follow-up questions → answer the new question; use tools if needed; don't dump the old answer again.
- Chat generally when the user is just talking — not every message needs a tool.

ANSWER FORMAT (for real questions):
- Neat markdown: short lead answer, then bullets/headings/tables if useful.
- Scannable, not a wall of text.
- A touch of personality is fine (e.g. "Good question!", "Here's the clean version:").
- If you used documents, add **Sources** with file names.
- If you used web search, briefly note **Based on web search**.
- After enough info, give the final answer only (no tool calls, no JSON).
"""

def _tools_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "Semantic search over the user's uploaded documents (knowledge base). "
                    "Use for questions about those files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    if ENABLE_WEB:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the public web for general knowledge or current information "
                        "not found in the uploaded documents."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Web search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    return tools


def _run_search_documents(query: str) -> str:
    hits = search_documents((query or "").strip())
    if not hits:
        return (
            "No documents matched this query. "
            "Knowledge base may be empty or unrelated — try web_search if available, "
            "or say you don't have that in the uploaded files."
        )
    return format_context(hits)


def _run_web_search(query: str) -> str:
    """Lightweight web search via DuckDuckGo HTML API (no key)."""
    try:
        import httpx
        from urllib.parse import quote_plus

        q = quote_plus((query or "").strip())
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "RAG-Agent/1.0"})
            r.raise_for_status()
            data = r.json()

        parts = []
        if data.get("AbstractText"):
            parts.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractURL"):
                parts.append(f"URL: {data['AbstractURL']}")
        related = data.get("RelatedTopics") or []
        for i, item in enumerate(related[:5], 1):
            if isinstance(item, dict) and item.get("Text"):
                parts.append(f"{i}. {item['Text']}")
                if item.get("FirstURL"):
                    parts.append(f"   Link: {item['FirstURL']}")
        if not parts:
            import re
            with httpx.Client(timeout=12.0, follow_redirects=True) as c2:
                lite = c2.get(
                    f"https://html.duckduckgo.com/html/?q={q}",
                    headers={"User-Agent": "Mozilla/5.0 RAG-Agent/1.0"},
                )
            text = lite.text
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', text, re.I | re.S)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</', text, re.I | re.S)
            def clean(s):
                return re.sub(r"<[^>]+>", "", s).strip()
            for i, title in enumerate(titles[:5], 1):
                parts.append(f"{i}. {clean(title)}")
                if i - 1 < len(snippets):
                    parts.append(f"   {clean(snippets[i-1])[:240]}")
        if not parts:
            return "Web search returned no useful results for that query."
        return "Web search results:\n" + "\n".join(parts)
    except Exception as e:
        return f"Web search failed: {e}"


def _collect_doc_sources(result: str, sources: list[str]) -> None:
    for line in result.splitlines():
        if "Source:" in line:
            src = line.split("Source:", 1)[-1].strip().split()[0]
            if src and src not in sources:
                sources.append(src)


async def agent_stream(
    user_message: str,
    history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> AsyncIterator[dict]:
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        yield {"type": "error", "message": "GROQ_API_KEY is not set."}
        return

    client = AsyncGroq(api_key=key)
    tools = _tools_schema()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    sources: list[str] = []
    used_web = False
    step_count = 0

    try:
        for step in range(MAX_AGENT_STEPS):
            step_count = step + 1
            yield {"type": "status", "message": f"Thinking (step {step_count})..."}

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_completion_tokens=1024,
            )
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    query = (args.get("query") or user_message).strip()

                    if name == "search_documents":
                        yield {"type": "tool", "query": f"[docs] {query}"}
                        result = _run_search_documents(query)
                        _collect_doc_sources(result, sources)
                    elif name == "web_search" and ENABLE_WEB:
                        yield {"type": "tool", "query": f"[web] {query}"}
                        result = _run_web_search(query)
                        used_web = True
                    else:
                        result = f"Tool '{name}' is not available."

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                continue

            # Final answer — stream with strict formatting reminder
            yield {"type": "status", "message": "Writing answer..."}
            stream = await client.chat.completions.create(
                model=model,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Write the final reply for the user now.\n"
                            "- If their message was only an acknowledgement (okay/got it/thanks/hi), "
                            "reply in 1 short friendly sentence — do NOT repeat prior answers.\n"
                            "- If it was a real question: neat markdown, short lead answer, then details.\n"
                            "- No tool calls. No JSON.\n"
                            "- Only cite documents from search_documents results; only claim web if web_search ran.\n"
                            "- Sound natural and a bit expressive, not robotic."
                        ),
                    }
                ],
                temperature=0.35,
                max_completion_tokens=2048,
                stream=True,
                tool_choice="none",
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield {"type": "delta", "content": delta}

            done_sources = list(sources)
            if used_web and "web_search" not in done_sources:
                done_sources.append("web_search")
            yield {"type": "done", "sources": done_sources, "steps": step_count}
            return

        yield {
            "type": "delta",
            "content": "I reached the step limit. Please ask a more specific question.",
        }
        yield {"type": "done", "sources": sources, "steps": step_count}

    except Exception as e:
        yield {"type": "error", "message": str(e)}
