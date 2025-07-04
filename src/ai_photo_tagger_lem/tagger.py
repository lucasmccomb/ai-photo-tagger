"""
AI Photo Tagger using OpenCLIP and Florence-2
"""

import os
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import logging
import re

from .config import Config
from .xmp_handler import XMPHandler


class PhotoTagger:
    """AI-powered photo tagger using OpenCLIP and Florence-2"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.florence_model = None
        self.florence_tokenizer = None
        self.vocab = []
        self.xmp_handler = XMPHandler()
        self.use_florence = self.config['model'].startswith('florence2://')
        
        # Initialize logger first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.use_florence:
            self._load_florence_model()
        else:
            self._load_model()
            self._load_vocab()
    
    def _get_device(self) -> torch.device:
        """Get the best available device (cuda -> mps -> cpu)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_florence_model(self):
        """Load Florence-2 model using transformers"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_id = self.config['model'].replace('florence2://', 'microsoft/Florence-2-')
            self.logger.info(f"Loading Florence-2 model: {model_id}")
            
            # Load tokenizer and model
            self.florence_tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type != "cuda":
                self.florence_model = self.florence_model.to(self.device)
            
            # Warn if on CPU with many images
            if self.device.type == 'cpu':
                self.logger.warning("Running Florence-2 on CPU - this will be slow for large batches")
            
            self.logger.info(f"Florence-2 model loaded successfully on {self.device}")
            
        except ImportError:
            raise ImportError("transformers not installed. Run: poetry add transformers")
        except Exception as e:
            self.logger.error(f"Failed to load Florence-2 model: {e}")
            raise
    
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
            # Filter out comments and empty lines, clean up tags
            self.vocab = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up the tag (remove extra spaces, ensure it's a valid tag)
                    tag = line.strip()
                    if tag and len(tag) <= 64:  # Respect 64 char limit
                        self.vocab.append(tag)
        
        self.logger.info(f"Loaded {len(self.vocab)} tags from vocabulary")
    
    def _get_image_files(self, directory: str) -> List[Path]:
        """Get all image files from directory and subdirectories recursively"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            # Search recursively with ** for all subdirectories
            image_files.extend(Path(directory).rglob(f"*{ext}"))
            image_files.extend(Path(directory).rglob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _preprocess_image(self, image_path: Path) -> torch.Tensor | None:
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocess is not None:
                # self.preprocess is a tuple of transforms, we need the first one
                transform = self.preprocess[0] if isinstance(self.preprocess, tuple) else self.preprocess
                # Convert PIL image to tensor properly
                tensor = transform(image)
                if isinstance(tensor, torch.Tensor):
                    return tensor.unsqueeze(0).to(self.device)
                else:
                    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
            return None
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def _get_text_embeddings(self) -> torch.Tensor:
        """Get text embeddings for vocabulary"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not initialized")
        
        # Use better prompting for more accurate classification
        prompted_vocab = [f"a photograph showing {tag}" for tag in self.vocab]
        text_tokens = self.tokenizer(prompted_vocab).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    

    
    def tag_image_florence2(self, image_path: Path) -> List[Tuple[str, float]]:
        """Tag a single image using Florence-2"""
        if self.florence_model is None or self.florence_tokenizer is None:
            raise RuntimeError("Florence-2 model not initialized")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Get prompt from config or use default
            prompt = self.config.get('florence_prompt', 'List key nouns in this image, comma separated.')
            
            # Prepare input for Florence-2
            inputs = self.florence_tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate tags using Florence-2
            with torch.no_grad():
                outputs = self.florence_model.generate(
                    **inputs,
                    max_new_tokens=40,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.florence_tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.florence_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the generated text to extract tags
            tags = self._parse_florence_output(generated_text)
            
            # Limit to top_k and add confidence scores
            top_k = self.config['clip_top_k']
            confidence_threshold = self.config['confidence_threshold']
            
            # Assign confidence scores (Florence-2 doesn't provide them, so we use a default)
            default_confidence = 0.8
            
            result_tags = []
            for i, tag in enumerate(tags[:top_k]):
                confidence = default_confidence - (i * 0.05)  # Slight confidence decay
                if confidence >= confidence_threshold:
                    result_tags.append((tag, confidence))
            
            return result_tags
            
        except Exception as e:
            self.logger.error(f"Error tagging image with Florence-2: {e}")
            return []
    
    def _parse_florence_output(self, generated_text: str) -> List[str]:
        """Parse Florence-2 output to extract clean tags"""
        # Remove the prompt from the beginning if present
        prompt = self.config.get('florence_prompt', 'List key nouns in this image, comma separated.')
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, '').strip()
        
        # Split by comma and clean up
        tags = []
        for tag in generated_text.split(','):
            tag = tag.strip().lower()
            # Remove common words and clean up
            tag = re.sub(r'[^\w\s-]', '', tag)  # Remove punctuation except hyphens
            tag = tag.strip()
            
            # Filter out empty tags and common stop words
            if tag and len(tag) > 1 and tag not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                tags.append(tag)
        
        return tags
    
    def tag_image(self, image_path: Path) -> List[Tuple[str, float]]:
        """Tag a single image (routes to appropriate model)"""
        if self.use_florence:
            return self.tag_image_florence2(image_path)
        else:
            return self._tag_image_openclip(image_path)
    
    def _tag_image_openclip(self, image_path: Path) -> List[Tuple[str, float]]:
        """Tag a single image using OpenCLIP (original implementation)"""
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
        
        # Calculate similarities (cosine similarity, range -1 to 1)
        similarity = image_features @ text_features.T
        
        # Get top predictions
        top_k = self.config['clip_top_k']
        confidence_threshold = self.config['confidence_threshold']
        
        values, indices = similarity[0].topk(top_k)
        
        # Debug: Print top predictions regardless of threshold
        self.logger.info(f"Top {top_k} predictions (before threshold):")
        for i, (value, idx) in enumerate(zip(values, indices)):
            confidence = value.item()
            confidence_normalized = (confidence + 1) / 2
            idx_int = int(idx.item())
            if 0 <= idx_int < len(self.vocab):
                tag = self.vocab[idx_int]
                self.logger.info(f"  {i+1}. {tag}: {confidence_normalized:.3f} (raw: {confidence:.3f})")
        
        tags = []
        for value, idx in zip(values, indices):
            confidence = value.item()
            # Convert from cosine similarity (-1 to 1) to confidence (0 to 1)
            confidence_normalized = (confidence + 1) / 2
            if confidence_normalized >= confidence_threshold:
                idx_int = int(idx.item())
                if 0 <= idx_int < len(self.vocab):
                    tag = self.vocab[idx_int]
                    tags.append((tag, confidence_normalized))
        
        return tags
    
    def process_directory(self) -> Dict[str, List[Tuple[str, float]]]:
        """Process all images in the configured directory with XMP and WebP export"""
        photos_dir = Path(self.config['photos_dir'])
        image_files = self._get_image_files(str(photos_dir))
        
        if not image_files:
            self.logger.warning(f"No image files found in {photos_dir}")
            return {}
        
        # Create tagged-jpeg-exports directory
        jpeg_exports_dir = photos_dir / "tagged-jpeg-exports"
        jpeg_exports_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created tagged-jpeg-exports directory: {jpeg_exports_dir}")
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = {}
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                self.logger.info(f"Processing: {image_path.name}")
                
                # Generate AI tags
                tags = self.tag_image(image_path)
                if tags:
                    results[str(image_path)] = tags
                    self.logger.info(f"AI tags: {[tag[0] for tag in tags]}")
                
                # Extract EXIF data
                exif_data = self.xmp_handler.get_exif_data(image_path)
                if exif_data:
                    self.logger.info(f"EXIF data: {len(exif_data)} tags extracted")
                
                # Check for existing XMP file
                xmp_path = image_path.with_suffix('.xmp')
                if xmp_path.exists():
                    self.logger.info(f"Found existing XMP file: {xmp_path.name}")
                    self.xmp_handler.update_xmp_file(xmp_path, tags, exif_data)
                else:
                    self.logger.info(f"Creating new XMP file for: {image_path.name}")
                    xmp_path = self.xmp_handler.create_xmp_file(image_path, tags, exif_data)
                
                # Log XMP data
                self.xmp_handler.log_xmp_data(xmp_path)
                
                # Export JPEG with embedded XMP data
                jpeg_path = self.xmp_handler.export_jpeg_with_xmp(
                    image_path, xmp_path, jpeg_exports_dir, max_size_mb=1.0
                )
                
                if jpeg_path:
                    self.logger.info(f"Exported JPEG: {jpeg_path.name}")
                else:
                    self.logger.error(f"Failed to export JPEG for: {image_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
        
        self.logger.info(f"Processing complete. JPEG exports saved to: {jpeg_exports_dir}")
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