"""
PageIndex Retrieval Service — reasoning-based search over a PageIndex tree.

Algorithm:
  1. Start at top-level nodes of the tree.
  2. Ask GPT: "Which of these sections most likely answers the query?"
  3. Recurse into selected branches.
  4. Return leaf / section text as context chunks.

This mirrors how a human expert navigates a document: scan the table of
contents, identify promising sections, dive into them.
"""
import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# Lazy OpenAI import
_openai_available = False
try:
    import openai
    _openai_available = True
except ImportError:
    logger.warning("openai package not installed. PageIndex retrieval unavailable.")


class PageIndexRetrieval:
    """
    Reasoning-based retrieval over a PageIndex tree structure.

    Uses a semaphore to limit concurrent OpenAI calls during tree traversal
    (multiple branches may be explored in parallel at each level).
    """

    def __init__(self):
        # The semaphore is shared with the indexing service via settings value
        self._semaphore = asyncio.Semaphore(settings.PAGEINDEX_MAX_CONCURRENT)

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return _openai_available and bool(settings.OPENAI_API_KEY.get_secret_value())

    async def search(
        self,
        query: str,
        tree_data: Dict[str, Any],
        top_k: int = 5,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Traverse the tree and return the top-k most relevant sections.

        Args:
            query:     User's question / search query
            tree_data: The PageIndex JSON (has 'doc_name' and 'structure' keys)
            top_k:     Maximum number of result sections to return
            max_depth: How deep to recurse into the tree (3 = great-grandchildren)

        Returns:
            List of section dicts with keys:
              file_id, filename, section_title, page_range, summary,
              text_preview, relevance_reasoning, similarity_score
        """
        if not self.available:
            raise RuntimeError(
                "PageIndex retrieval unavailable. Check OPENAI_API_KEY and openai package."
            )

        structure = tree_data.get("structure", [])
        doc_name = tree_data.get("doc_name", "Document")

        if not structure:
            return []

        collected: List[Dict[str, Any]] = []
        await self._traverse(
            query=query,
            nodes=structure,
            doc_name=doc_name,
            collected=collected,
            depth=0,
            max_depth=max_depth,
        )

        # Sort by relevance score (LLM-assigned, 0-1) and return top_k
        collected.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return collected[:top_k]

    async def generate_answer(
        self,
        question: str,
        context_sections: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a final answer using OpenAI GPT from the retrieved sections.

        Args:
            question:          The user's question
            context_sections:  Sections returned by search()
            conversation_history: Optional prior conversation turns

        Returns:
            Dict with 'answer', 'sources', 'confidence', 'tokens_used'
        """
        if not self.available:
            return {
                "answer": "PageIndex retrieval unavailable. Please configure OPENAI_API_KEY.",
                "sources": [],
                "confidence": 0.0,
                "tokens_used": 0,
            }

        if not context_sections:
            return {
                "answer": "No relevant sections found in the indexed documents.",
                "sources": [],
                "confidence": 0.0,
                "tokens_used": 0,
            }

        # Build context string from retrieved sections
        context_parts = []
        for i, sec in enumerate(context_sections[:5], 1):
            title = sec.get("section_title", "Section")
            pages = sec.get("page_range", "")
            preview = sec.get("text_preview", "")
            summary = sec.get("summary", "")
            doc = sec.get("filename", "Document")
            context_parts.append(
                f"[Source {i}: {doc} — {title}"
                + (f" (pages {pages})" if pages else "")
                + f"]\n"
                + (f"Summary: {summary}\n" if summary else "")
                + (f"Content: {preview}" if preview else "")
            )

        context_str = "\n\n".join(context_parts)

        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. Answer the user's question "
                    "using ONLY the provided document sections. Cite sources by their "
                    "Source number. If the sections don't contain enough information, "
                    "say so clearly and concisely."
                ),
            }
        ]

        # Add a few turns of conversation history (for multi-turn context)
        if conversation_history:
            for turn in conversation_history[-3:]:
                messages.append({"role": "user", "content": turn.get("question", "")})
                messages.append({"role": "assistant", "content": turn.get("answer", "")})

        messages.append({
            "role": "user",
            "content": (
                f"Document sections:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Provide a clear, well-cited answer."
            ),
        })

        try:
            async with self._semaphore:
                response = await self._call_openai_chat(messages, max_tokens=1024)

            answer = response["content"]
            tokens_used = response.get("tokens_used", 0)

            sources = [
                {
                    "filename": s.get("filename", "Unknown"),
                    "section_title": s.get("section_title", ""),
                    "page_range": s.get("page_range", ""),
                    "similarity_score": round(s.get("similarity_score", 0.0), 3),
                    "preview": (s.get("text_preview") or s.get("summary", ""))[:200],
                }
                for s in context_sections[:5]
            ]

            # Confidence heuristic: average similarity score, capped at 0.95
            avg_sim = (
                sum(s.get("similarity_score", 0) for s in context_sections[:3])
                / min(3, len(context_sections))
                if context_sections
                else 0
            )
            confidence = min(0.95, avg_sim)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 3),
                "tokens_used": tokens_used,
            }

        except Exception as e:
            logger.error(f"PageIndex answer generation failed: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "tokens_used": 0,
            }

    # ------------------------------------------------------------------ #
    #  Tree traversal (private)                                            #
    # ------------------------------------------------------------------ #

    async def _traverse(
        self,
        query: str,
        nodes: List[Dict],
        doc_name: str,
        collected: List[Dict],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively explore tree nodes, asking GPT which branches to follow."""
        if not nodes or depth > max_depth:
            return

        # Decide which nodes to explore at this level
        relevant_nodes, reasonings = await self._select_relevant_nodes(query, nodes)

        tasks = []
        for node, reasoning in zip(relevant_nodes, reasonings):
            tasks.append(
                self._process_node(
                    query=query,
                    node=node,
                    doc_name=doc_name,
                    collected=collected,
                    depth=depth,
                    max_depth=max_depth,
                    reasoning=reasoning,
                )
            )
        await asyncio.gather(*tasks)

    async def _process_node(
        self,
        query: str,
        node: Dict,
        doc_name: str,
        collected: List[Dict],
        depth: int,
        max_depth: int,
        reasoning: str,
    ) -> None:
        """Process a single selected node: collect it and recurse if it has children."""
        children = node.get("nodes", [])

        # Compute a relevance score (0-1) from the reasoning verdict
        score = self._extract_score_from_reasoning(reasoning)

        section = {
            "filename": doc_name,
            "section_title": node.get("title", ""),
            "node_id": node.get("node_id", ""),
            "page_range": self._format_page_range(node),
            "summary": node.get("summary", ""),
            "text_preview": (node.get("text", "") or node.get("summary", ""))[:500],
            "relevance_reasoning": reasoning,
            "similarity_score": score,
        }
        collected.append(section)

        # Recurse into children if not at max depth
        if children and depth < max_depth:
            await self._traverse(
                query=query,
                nodes=children,
                doc_name=doc_name,
                collected=collected,
                depth=depth + 1,
                max_depth=max_depth,
            )

    async def _select_relevant_nodes(
        self, query: str, nodes: List[Dict]
    ) -> tuple[List[Dict], List[str]]:
        """
        Ask GPT which nodes in this list are relevant to the query.
        Returns (relevant_nodes, reasonings).
        """
        if not nodes:
            return [], []

        # Build a compact summary of each node for the prompt
        node_descriptions = []
        for i, node in enumerate(nodes):
            title = node.get("title", f"Section {i+1}")
            summary = node.get("summary", "")
            desc = f"{i+1}. {title}"
            if summary:
                desc += f": {summary[:200]}"
            node_descriptions.append(desc)

        prompt_content = (
            "You are helping navigate a document to answer a question.\n"
            "Select all sections that are LIKELY to contain information relevant to the question.\n"
            "You may select multiple sections. Be inclusive rather than exclusive.\n\n"
            f"Question: {query}\n\n"
            "Available sections:\n"
            + "\n".join(node_descriptions)
            + "\n\n"
            "Reply ONLY with a JSON array. Each element must be:\n"
            '{"index": <1-based number>, "relevant": true/false, '
            '"score": <0.0-1.0 relevance score>, "reason": "<one sentence>"}\n'
            "Include ALL sections in the reply, marking non-relevant ones as false."
        )

        messages = [{"role": "user", "content": prompt_content}]

        try:
            async with self._semaphore:
                response = await self._call_openai_chat(messages, max_tokens=512)

            raw = response["content"]
            verdicts = self._parse_json_response(raw)

            relevant_nodes = []
            reasonings = []
            for verdict in verdicts:
                idx = verdict.get("index", 0) - 1  # convert to 0-based
                if not isinstance(idx, int) or idx < 0 or idx >= len(nodes):
                    continue
                if verdict.get("relevant", False):
                    relevant_nodes.append(nodes[idx])
                    score = verdict.get("score", 0.5)
                    reason = verdict.get("reason", "")
                    reasonings.append(f"Score:{score:.2f} — {reason}")

            # Fallback: if GPT selected nothing, take all nodes with minimal score
            if not relevant_nodes:
                relevant_nodes = nodes
                reasonings = ["fallback: no specific match found"] * len(nodes)

            return relevant_nodes, reasonings

        except Exception as e:
            logger.warning(f"Node selection failed (returning all nodes as fallback): {e}")
            return nodes, ["fallback"] * len(nodes)

    # ------------------------------------------------------------------ #
    #  OpenAI helpers                                                       #
    # ------------------------------------------------------------------ #

    async def _call_openai_chat(
        self, messages: List[Dict], max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Async wrapper around the OpenAI Chat Completions API with retry logic."""
        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=settings.PAGEINDEX_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                tokens_used = resp.usage.total_tokens if resp.usage else 0
                return {"content": content, "tokens_used": tokens_used}

            except openai.RateLimitError as e:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                logger.warning(f"OpenAI rate limit hit (attempt {attempt+1}). Waiting {wait}s.")
                await asyncio.sleep(wait)

            except openai.APIError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise

        raise RuntimeError("OpenAI API call failed after max retries.")

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json_response(raw: str) -> List[Dict]:
        """Extract JSON array from GPT response."""
        import re
        # Strip markdown fences if present
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            raw = match.group(1)
        try:
            data = json.loads(raw.strip())
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from GPT response: {raw[:200]}")
            return []

    @staticmethod
    def _format_page_range(node: Dict) -> str:
        start = node.get("start_index")
        end = node.get("end_index")
        if start and end:
            return f"{start}–{end}"
        if start:
            return str(start)
        return ""

    @staticmethod
    def _extract_score_from_reasoning(reasoning: str) -> float:
        """Parse score from 'Score:0.85 — ...' format or default to 0.5."""
        import re
        m = re.search(r"Score:([\d.]+)", reasoning)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.5


# ---- Global singleton ----
page_index_retrieval = PageIndexRetrieval()
