"""SIE Router - Stateless request router for elastic cloud deployments.

The router provides:
- Load balancing across multiple workers
- Routing by model + GPU type
- Worker state tracking via WebSocket connections
- 202 responses for provisioning delays
- Cluster-wide status aggregation
"""

__version__ = "0.1.0"
