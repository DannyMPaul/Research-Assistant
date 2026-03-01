"""
PageIndex Service — wraps the PageIndex library for document tree indexing.

Responsibilities:
- Index a PDF document → produces a hierarchical tree JSON (table-of-contents style)
- Persist the tree JSON to disk (page_index_store/{file_id}.json)
- Load / delete / list indexed documents
- Rate-control concurrent OpenAI calls via asyncio.Semaphore
"""
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

# Patch the env var that the PageIndex library reads for OpenAI key BEFORE import
# The library reads CHATGPT_API_KEY from os.environ inside utils.py
# We forward OPENAI_API_KEY → CHATGPT_API_KEY so nothing is hardcoded
from config import settings

logger = logging.getLogger(__name__)

# Lazy-imports so the service is importable even if optional deps aren't installed
_pageindex_available = False
try:
    from types import SimpleNamespace as _config
    from pageindex.page_index import page_index_main as _pi_main
    from pageindex.utils import ConfigLoader as _ConfigLoader
    _pageindex_available = True
    logger.info("PageIndex library loaded successfully.")
except ImportError as _e:
    logger.warning(f"PageIndex library not available: {_e}. Install required deps.")


class PageIndexService:
    """Service that manages document indexing using the PageIndex approach."""

    def __init__(self):
        self.store_dir = settings.PAGEINDEX_STORE_DIR
        self.store_dir.mkdir(exist_ok=True)
        # Semaphore to limit concurrent OpenAI calls during indexing
        self._semaphore = asyncio.Semaphore(settings.PAGEINDEX_MAX_CONCURRENT)

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return _pageindex_available and bool(settings.OPENAI_API_KEY.get_secret_value())

    def document_indexed(self, file_id: str) -> bool:
        """Check whether a document has already been indexed."""
        return self._index_path(file_id).exists()

    def get_index(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load the tree JSON for a document, or None if not indexed."""
        path = self._index_path(file_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading page index for {file_id}: {e}")
            return None

    def delete_index(self, file_id: str) -> bool:
        """Remove the stored tree JSON. Returns True if deleted."""
        path = self._index_path(file_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted page index for {file_id}")
            return True
        return False

    def list_indexed(self) -> List[Dict[str, Any]]:
        """Return metadata for all indexed documents."""
        results = []
        for p in self.store_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                results.append({
                    "file_id": p.stem,
                    "doc_name": data.get("doc_name", p.stem),
                    "doc_description": data.get("doc_description", ""),
                    "node_count": self._count_nodes(data.get("structure", [])),
                })
            except Exception as e:
                logger.warning(f"Could not read index file {p}: {e}")
        return results

    async def index_document(self, file_id: str, file_path: str) -> Dict[str, Any]:
        """
        Build a PageIndex tree for the given PDF file and save it to disk.

        Args:
            file_id:   UUID stem of the uploaded file (used as storage key)
            file_path: Absolute path to the PDF file on disk

        Returns:
            Dictionary with indexing result info.

        Raises:
            RuntimeError: if PageIndex is unavailable or the file is not a PDF
            FileNotFoundError: if the file doesn't exist
        """
        if not self.available:
            raise RuntimeError(
                "PageIndex is not available. Ensure the pageindex library is installed "
                "and OPENAI_API_KEY is set in your .env file."
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PageIndex only supports PDF files. Got: {path.suffix}")

        logger.info(f"Starting PageIndex indexing for file_id={file_id} path={file_path}")

        try:
            # Run the synchronous page_index_main in a thread pool so we don't
            # block the event loop. The semaphore limits total concurrent indexing jobs.
            async with self._semaphore:
                tree_data = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._run_indexing,
                    str(path),
                )

            # Persist to disk
            out_path = self._index_path(file_id)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tree_data, f, indent=2, ensure_ascii=False)

            node_count = self._count_nodes(tree_data.get("structure", []))
            logger.info(
                f"PageIndex complete for {file_id}: {node_count} nodes saved to {out_path}"
            )
            return {
                "file_id": file_id,
                "doc_name": tree_data.get("doc_name", path.name),
                "node_count": node_count,
                "status": "indexed",
            }

        except Exception as e:
            logger.error(f"PageIndex indexing failed for {file_id}: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_indexing(self, pdf_path: str) -> Dict[str, Any]:
        config_loader = _ConfigLoader()
        opt = config_loader.load({
            "model": settings.PAGEINDEX_MODEL,
            "toc_check_page_num": settings.PAGEINDEX_TOC_CHECK_PAGES,
            "max_page_num_each_node": settings.PAGEINDEX_MAX_PAGES_PER_NODE,
            "max_token_num_each_node": settings.PAGEINDEX_MAX_TOKENS_PER_NODE,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_doc_description": "no",
            "if_add_node_text": "no",
        })
        return _pi_main(pdf_path, opt)

    def _index_path(self, file_id: str) -> Path:
        return self.store_dir / f"{file_id}.json"

    def _count_nodes(self, structure) -> int:
        """Recursively count all nodes in the tree."""
        if isinstance(structure, list):
            return sum(self._count_nodes(n) for n in structure)
        if isinstance(structure, dict):
            children = structure.get("nodes", [])
            return 1 + self._count_nodes(children)
        return 0


# ---- Global singleton ----
page_index_service = PageIndexService()
