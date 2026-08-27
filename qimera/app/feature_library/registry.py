"""QiMera Metadata Registry Manager (QiMera.md §6.3 / M1).

Provides an API to read, query, and register features dynamically in the
Feature Library's Metadata Registry manifest.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

REGISTRY_PATH = Path(__file__).resolve().parent / "metadata_registry.json"


class MetadataRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self._registry_data = self._load()

    def _load(self) -> dict:
        if not self.path.is_file():
            # Return empty skeleton
            return {"schema_version": 1, "features": []}
            
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return {"schema_version": 1, "features": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._registry_data, f, indent=2, ensure_ascii=False)

    def list_features(self, feature_type: Optional[str] = None) -> List[Dict]:
        """Returns all registered features, optionally filtered by type (primitive/building_block)."""
        features = self._registry_data.get("features", [])
        if feature_type:
            return [f for f in features if f.get("type") == feature_type]
        return features

    def get_feature(self, feature_id: str) -> Optional[Dict]:
        """Retrieves a single feature definition by its unique identifier."""
        for f in self._registry_data.get("features", []):
            if f.get("id") == feature_id:
                return f
        return None

    def register_feature(self, feature_id: str, name: str, feature_type: str,
                         function_name: str, parameters: dict,
                         description: str, provenance: Optional[dict] = None) -> Dict:
        """Dynamically adds a new feature primitive/building block to the manifest.
        
        Typically called by the Indicator Interpretation Agent.
        """
        # Enforce unique IDs
        if self.get_feature(feature_id):
            raise ValueError(f"Feature with ID '{feature_id}' is already registered.")
            
        new_entry = {
            "id": feature_id,
            "name": name,
            "type": feature_type,
            "function_name": function_name,
            "parameters": parameters,
            "output_range": [-1.0, 1.0],
            "description": description,
            "provenance": provenance
        }
        
        self._registry_data.setdefault("features", []).append(new_entry)
        self._save()
        return new_entry
