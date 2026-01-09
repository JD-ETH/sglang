"""
Cross-node request forwarding for remote instance transfer engine info.

This module handles forwarding requests to remote nodes when the requested
tp_rank is not available locally.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from http import HTTPStatus

try:
    import aiohttp
except ImportError:
    aiohttp = None

from sglang.srt.model_loader.node_registry import NodeInfo

logger = logging.getLogger(__name__)


class CrossNodeForwarder:
    """Handles forwarding requests to remote nodes."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for cross-node forwarding. Install with: pip install aiohttp")

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def forward_get_remote_instance_transfer_engine_info(
        self,
        target_node: NodeInfo,
        rank: int
    ) -> Dict[str, Any]:
        """
        Forward get_remote_instance_transfer_engine_info request to target node.

        Args:
            target_node: Node to forward the request to
            rank: The tp_rank to query

        Returns:
            Response from the target node

        Raises:
            Exception: If forwarding fails after all retries
        """
        url = f"http://{target_node.host}:{target_node.port}/get_remote_instance_transfer_engine_info"
        params = {"rank": rank % 8}

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Forwarding request for rank {rank} to node {target_node.node_rank} "
                           f"at {target_node.host}:{target_node.port} (attempt {attempt + 1}/{self.max_retries})")

                session = await self._get_session()

                async with session.get(url, params=params) as response:
                    if response.status == HTTPStatus.OK:
                        result = await response.json()
                        logger.info(f"Successfully received response for rank {rank} from node {target_node.node_rank}")
                        return result
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Remote node returned status {response.status}: {error_text}"
                        )

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for rank {rank}: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 0.5 * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for rank {rank}")

        # All retries failed
        raise Exception(f"Failed to forward request for rank {rank} to node {target_node.node_rank}: {last_exception}")

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global forwarder instance
_global_forwarder: Optional[CrossNodeForwarder] = None


def get_global_forwarder() -> CrossNodeForwarder:
    """Get the global forwarder instance."""
    global _global_forwarder
    if _global_forwarder is None:
        _global_forwarder = CrossNodeForwarder()
    return _global_forwarder


async def shutdown_global_forwarder():
    """Shutdown the global forwarder."""
    global _global_forwarder
    if _global_forwarder is not None:
        await _global_forwarder.close()
        _global_forwarder = None