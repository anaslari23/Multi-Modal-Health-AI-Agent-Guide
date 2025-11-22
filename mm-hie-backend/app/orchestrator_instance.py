from __future__ import annotations

"""Shared orchestrator instance for all API routers.

This avoids circular imports between app.main and routers that need
access to the same Orchestrator object.
"""

from .orchestrator import Orchestrator

orchestrator = Orchestrator()
