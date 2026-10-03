"""
Backend Configuration for E-Dermatologist
"""

import os
from pathlib import Path
import yaml
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


BASE_DIR = Path(__file__).resolve().parent.parent.parent

def _load_env_file():
    for env_path in [BASE_DIR / ".env", BASE_DIR / "backend" / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()
CONFIG_PATH = os.getenv("CLASSES_PATH", str(BASE_DIR / "configs" / "classes.yaml"))


class AppConfig:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.classes: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {}
        self.modality: str = "mobile_spacer"
        self.version: str = "v2.0-mobile"
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.classes = data.get("classes", [])
                self.thresholds = data.get("thresholds", {})
                self.modality = data.get("modality", "mobile_spacer")
                self.version = data.get("version", "v2.0-mobile")
        else:
            # Fallback defaults
            self.classes = [
                {"id": "eczema", "label": "Eczema", "severity": "common", "color": "#2563EB"},
                {"id": "psoriasis", "label": "Psoriasis", "severity": "common", "color": "#D97706"},
                {"id": "tinea", "label": "Tinea (Ringworm)", "severity": "common", "color": "#7C3AED"},
                {"id": "acne", "label": "Acne", "severity": "common", "color": "#059669"},
                {"id": "healthy", "label": "Healthy Skin", "severity": "normal", "color": "#10B981"},
                {"id": "suspicious_lesion", "label": "Suspicious Lesion", "severity": "urgent", "color": "#DC2626"}
            ]
            self.thresholds = {
                "uncertain_max_prob": 0.50,
                "margin_min_prob": 0.10,
                "urgent_suspicious_prob": 0.25,
                "blur_min_laplacian_var": 55.0,
                "brightness_range": [40, 225],
                "glare_max_ratio": 0.12
            }


config = AppConfig()
