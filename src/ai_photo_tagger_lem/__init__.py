"""
AI Photo Tagger - Automated photo tagging using OpenCLIP
"""

__version__ = "0.1.0"

from .tagger import PhotoTagger
from .config import Config
from .xmp_handler import XMPHandler

__all__ = ["PhotoTagger", "Config", "XMPHandler"]
