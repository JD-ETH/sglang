"""
Node Registry for Cross-Node Communication

This module provides functionality to map tp_ranks to their hosting nodes
and facilitate cross-node communication for remote instance transfer engine info.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a node in the distributed setup."""
    node_rank: int
    host: str
    port: int
    tp_rank_range: range

    def __contains__(self, tp_rank: int) -> bool:
        """Check if this node hosts the given tp_rank."""
        return tp_rank in self.tp_rank_range


class NodeRegistry:
    """Registry for mapping tp_ranks to nodes in a distributed setup."""

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.nodes: Dict[int, NodeInfo] = {}
        self._populate_node_mapping()

    def _populate_node_mapping(self):
        """Build node registry using same logic as scheduler_launcher.py"""
        # Same calculation logic as in scheduler_launcher.py:90-95
        nnodes_per_tp_group = max(self.server_args.nnodes // self.server_args.pp_size, 1)
        tp_size_per_node = self.server_args.tp_size // nnodes_per_tp_group

        logger.info(f"Building node registry: nnodes={self.server_args.nnodes}, "
                   f"tp_size={self.server_args.tp_size}, pp_size={self.server_args.pp_size}")
        logger.info(f"Calculated: nnodes_per_tp_group={nnodes_per_tp_group}, "
                   f"tp_size_per_node={tp_size_per_node}")

        for node_rank in range(self.server_args.nnodes):
            tp_start = tp_size_per_node * (node_rank % nnodes_per_tp_group)
            tp_end = tp_size_per_node * (node_rank % nnodes_per_tp_group + 1)

            host, port = self._get_node_host_port(node_rank)

            node_info = NodeInfo(
                node_rank=node_rank,
                host=host,
                port=port,
                tp_rank_range=range(tp_start, tp_end)
            )

            self.nodes[node_rank] = node_info
            logger.info(f"Node {node_rank}: host={host}:{port}, tp_ranks={list(node_info.tp_rank_range)}")

    def _get_node_host_port(self, node_rank: int) -> tuple[str, int]:
        """
        Determine host and port for a given node_rank.

        Implementation options:
        1. Use server_args configuration
        2. Use distributed initialization addresses
        3. Use environment variables or config files
        """
        # Option 1: Use server_args host/port with offset
        base_host = getattr(self.server_args, 'host', '127.0.0.1')
        base_port = getattr(self.server_args, 'port', 30000)

        # For multi-node setups, we need to determine the actual hosts
        # This is a simplified approach - in practice, you might want to:
        # - Use a configuration file with explicit node addresses
        # - Use service discovery
        # - Parse from distributed initialization addresses

        if hasattr(self.server_args, 'node_hosts') and self.server_args.node_hosts:
            # Option: Custom node_hosts configuration
            # Format: "host1:port1,host2:port2,..."
            node_addresses = self.server_args.node_hosts.split(',')
            if node_rank < len(node_addresses):
                host_port = node_addresses[node_rank].strip()
                if ':' in host_port:
                    host, port_str = host_port.split(':', 1)
                    return host.strip(), int(port_str.strip())
                else:
                    return host_port.strip(), base_port

        # Fallback: Calculate based on node_rank
        # This assumes all nodes use sequential ports on the same host
        # In real multi-node setups, you'd need actual host discovery
        if node_rank == 0:
            return base_host, base_port
        else:
            # For demonstration: assume nodes use different ports
            # In practice, different nodes would have different hosts
            return base_host, base_port + node_rank * 1000

    def get_node_for_rank(self, tp_rank: int) -> Optional[NodeInfo]:
        """Get the node info for a given tp_rank."""
        for node_info in self.nodes.values():
            if tp_rank in node_info:
                return node_info
        return None

    def get_local_tp_ranks(self) -> List[int]:
        """Get all tp_ranks hosted by the current node."""
        current_node_rank = self.server_args.node_rank
        if current_node_rank in self.nodes:
            return list(self.nodes[current_node_rank].tp_rank_range)
        return []

    def is_local_rank(self, tp_rank: int) -> bool:
        """Check if the given tp_rank is local to this node."""
        current_node_rank = self.server_args.node_rank
        if current_node_rank in self.nodes:
            return tp_rank in self.nodes[current_node_rank]
        return False

    def get_all_nodes(self) -> Dict[int, NodeInfo]:
        """Get all nodes in the registry."""
        return self.nodes.copy()

    def validate_tp_rank(self, tp_rank: int) -> bool:
        """Validate if the tp_rank is within the valid range."""
        return 0 <= tp_rank < self.server_args.tp_size


def create_node_registry(server_args: ServerArgs) -> NodeRegistry:
    """Factory function to create a node registry."""
    return NodeRegistry(server_args)