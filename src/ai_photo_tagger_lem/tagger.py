"""
AI Photo Tagger using OpenCLIP
"""

import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import logging

from .config import Config


class PhotoTagger:
    """AI-powered photo tagger using OpenCLIP"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.vocab = []
        
        self._load_model()
        self._load_vocab()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load OpenCLIP model"""
        model_name = self.config['model'].replace('openclip://', '')
        self.logger.info(f"Loading model: {model_name}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            device=self.device,
            precision='fp32'
        )
        
        # Get tokenizer separately
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def _load_vocab(self):
        """Load vocabulary from file"""
        vocab_path = self.config['clip_vocab']
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path, 'r') as f:
            self.vocab = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Loaded {len(self.vocab)} tags from vocabulary")
    
    def _get_image_files(self, directory: str) -> List[Path]:
        """Get all image files from directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _preprocess_image(self, image_path: Path) -> torch.Tensor | None:
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess is not None:
                # self.preprocess is a tuple of transforms, we need the first one
                transform = self.preprocess[0] if isinstance(self.preprocess, tuple) else self.preprocess
                return torch.tensor(transform(image)).unsqueeze(0).to(self.device)
            return None
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def _get_text_embeddings(self) -> torch.Tensor:
        """Get text embeddings for vocabulary"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")
        text_tokens = self.tokenizer(self.vocab).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def tag_image(self, image_path: Path) -> List[Tuple[str, float]]:
        """Tag a single image"""
        # Preprocess image
        image_tensor = self._preprocess_image(image_path)
        if image_tensor is None:
            return []
        
        # Get text embeddings
        text_features = self._get_text_embeddings()
        
        # Get image features
        if self.model is None:
            raise RuntimeError("Model not initialized")
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get top predictions
        top_k = self.config['clip_top_k']
        confidence_threshold = self.config['confidence_threshold']
        
        values, indices = similarity[0].topk(top_k)
        
        tags = []
        for value, idx in zip(values, indices):
            confidence = value.item()
            if confidence >= confidence_threshold:
                idx_int = int(idx.item())
                if 0 <= idx_int < len(self.vocab):
                    tag = self.vocab[idx_int]
                    tags.append((tag, confidence))
        
        return tags
    
    def process_directory(self) -> Dict[str, List[Tuple[str, float]]]:
        """Process all images in the configured directory"""
        photos_dir = self.config['photos_dir']
        image_files = self._get_image_files(photos_dir)
        
        if not image_files:
            self.logger.warning(f"No image files found in {photos_dir}")
            return {}
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = {}
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                tags = self.tag_image(image_path)
                if tags:
                    results[str(image_path)] = tags
                    self.logger.info(f"{image_path.name}: {[tag[0] for tag in tags]}")
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
        
        return results
    
    def save_results(self, results: Dict[str, List[Tuple[str, float]]], output_format: str | None = None):
        """Save tagging results"""
        if output_format is None:
            output_format = self.config.get('output_format', 'xmp')
        
        if output_format == 'xmp':
            self._save_xmp(results)
        elif output_format == 'json':
            self._save_json(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_xmp(self, results: Dict[str, List[Tuple[str, float]]]):
        """Save results as XMP metadata"""
        # This is a placeholder - XMP writing would require additional libraries
        # like exiftool or similar
        self.logger.info("XMP output format not yet implemented")
        self.logger.info("Results would be saved as XMP metadata")
    
    def _save_json(self, results: Dict[str, List[Tuple[str, float]]]):
        """Save results as JSON file"""
        import json
        
        output_file = "photo_tags.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}") 