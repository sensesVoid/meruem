"""Re-export entities so SQLModel registers all tables on import.

Importing ``app.models`` (e.g. ``from app.models import entities`` in db.py)
triggers this module, which loads every table class. The re-exports are for
convenience; each name is intentionally unused in this module.
"""
from app.models.entities import (  # noqa: F401
    Account,
    Asset,
    LTRRecord,
    Scan,
    Setting,
    Strategy,
    Trade,
    Validation,
)