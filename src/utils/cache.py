import hashlib
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.utils import logging_mod

logger = logging_mod.get_logging(__name__)

def generate_identifier(instance_data: Dict[str, Any]) -> str:
    """
    Creates a unique MD5 signature for the current financial problem.
    We sort keys to ensure the hash is deterministic.
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for k, v in instance_data.items():
        if isinstance(v, (np.ndarray, list)):
            # We hash the mean/std to save string space if the array is huge
            serializable_data[k] = hashlib.sha256(np.array(v).tobytes()).hexdigest()
        else:
            serializable_data[k] = v
    
    data_str = json.dumps(serializable_data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


class LandscapeCacheProxy:
    def __init__(self, cache_dir: str = "artifacts"):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_path(self, data_hash: str, p: int) -> Path:
        return self.root / f"{data_hash}_p{p}.pkl"

    def exists(self, data_hash: str, p: int) -> bool:
        return self._get_path(data_hash, p).exists()

    def save(self, data_hash: str, p: int, angles: np.ndarray):
        path = self._get_path(data_hash, p)
        with open(path, "wb") as f:
            pickle.dump(angles, f)
        logger.info(f"--- Cache Saved: {path.name} ---")

    def load(self, data_hash: str, p: int) -> Optional[np.ndarray]:
        path = self._get_path(data_hash, p)
        
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            logger.info(f"--- Cache Loaded: {path.name} ---")
            return pickle.load(f)
        