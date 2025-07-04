"""
Configuration management for AI Photo Tagger
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for AI Photo Tagger"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate configuration values"""
        # Always required
        required_fields = ['photos_dir', 'model', 'clip_top_k', 'confidence_threshold']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Only require clip_vocab for OpenCLIP
        if self.config['model'].startswith('openclip://'):
            if 'clip_vocab' not in self.config:
                raise ValueError("Missing required configuration field: clip_vocab (required for OpenCLIP)")

        # Expand user path for photos_dir
        self.config['photos_dir'] = os.path.expanduser(self.config['photos_dir'])
        # Validate photos directory exists
        if not os.path.exists(self.config['photos_dir']):
            raise ValueError(f"Photos directory does not exist: {self.config['photos_dir']}")
        # Validate confidence threshold
        if not 0 <= self.config['confidence_threshold'] <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        # Validate top_k
        if self.config['clip_top_k'] <= 0:
            raise ValueError("clip_top_k must be positive")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self.config 