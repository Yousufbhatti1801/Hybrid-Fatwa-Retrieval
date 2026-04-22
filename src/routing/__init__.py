"""Smart-router package — escalating retrieval across three paths.

See :mod:`src.routing.router` for the orchestrator and the per-stage
confidence scorers.  Consumers should only need ``route_query``.
"""

from __future__ import annotations

from src.routing.router import route_query

__all__ = ["route_query"]
