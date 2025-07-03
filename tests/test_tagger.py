"""
Tests for AI Photo Tagger
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from ai_photo_tagger_lem.config import Config
from ai_photo_tagger_lem.tagger import PhotoTagger


class TestConfig:
    """Test configuration loading"""
    
    def test_valid_config(self):
        """Test loading valid configuration"""
        config_data = """
photos_dir: ~/Pictures/test
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 0.2
output_format: json
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            config = Config(config_path)
            assert config['photos_dir'] == os.path.expanduser('~/Pictures/test')
            assert config['model'] == 'openclip://ViT-B-32'
            assert config['clip_top_k'] == 5
            assert config['confidence_threshold'] == 0.2
        finally:
            os.unlink(config_path)
    
    def test_missing_required_field(self):
        """Test error when required field is missing"""
        config_data = """
photos_dir: ~/Pictures/test
model: openclip://ViT-B-32
# clip_vocab missing
clip_top_k: 5
confidence_threshold: 0.2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required configuration field"):
                Config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_invalid_confidence_threshold(self):
        """Test error when confidence threshold is invalid"""
        config_data = """
photos_dir: ~/Pictures/test
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 1.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="confidence_threshold must be between 0 and 1"):
                Config(config_path)
        finally:
            os.unlink(config_path)


class TestPhotoTagger:
    """Test photo tagger functionality"""
    
    @patch('ai_photo_tagger_lem.tagger.open_clip')
    def test_tagger_initialization(self, mock_open_clip):
        """Test tagger initialization"""
        # Mock OpenCLIP components
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_tokenizer = Mock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model, mock_preprocess, mock_tokenizer
        )
        
        # Create temporary config and vocab files
        config_data = """
photos_dir: ~/Pictures/test
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 0.2
output_format: json
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            config_path = f.name
        
        vocab_data = "person\ndog\ncat\nmountain"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(vocab_data)
            vocab_path = f.name
        
        try:
            # Update config to use our vocab file
            config = Config(config_path)
            config.config['clip_vocab'] = vocab_path
            
            # Create tagger
            tagger = PhotoTagger(config)
            
            assert tagger.model == mock_model
            assert tagger.preprocess == mock_preprocess
            assert tagger.tokenizer == mock_tokenizer
            assert len(tagger.vocab) == 4
            assert 'person' in tagger.vocab
            assert 'dog' in tagger.vocab
        
        finally:
            os.unlink(config_path)
            os.unlink(vocab_path)
    
    def test_get_image_files(self):
        """Test finding image files in directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_files = [
                'photo1.jpg',
                'photo2.png',
                'document.txt',
                'photo3.JPEG',
                'photo4.tiff'
            ]
            
            for filename in test_files:
                (Path(temp_dir) / filename).touch()
            
            # Create config pointing to temp directory
            config_data = f"""
photos_dir: {temp_dir}
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 0.2
output_format: json
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(config_data)
                config_path = f.name
            
            vocab_data = "person\ndog"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(vocab_data)
                vocab_path = f.name
            
            try:
                config = Config(config_path)
                config.config['clip_vocab'] = vocab_path
                
                with patch('ai_photo_tagger_lem.tagger.open_clip') as mock_open_clip:
                    mock_model = Mock()
                    mock_preprocess = Mock()
                    mock_tokenizer = Mock()
                    mock_open_clip.create_model_and_transforms.return_value = (
                        mock_model, mock_preprocess, mock_tokenizer
                    )
                    
                    tagger = PhotoTagger(config)
                    image_files = tagger._get_image_files(temp_dir)
                    
                    # Should find 4 image files (jpg, png, JPEG, tiff)
                    assert len(image_files) == 4
                    file_names = [f.name for f in image_files]
                    assert 'photo1.jpg' in file_names
                    assert 'photo2.png' in file_names
                    assert 'photo3.JPEG' in file_names
                    assert 'photo4.tiff' in file_names
                    assert 'document.txt' not in file_names
            
            finally:
                os.unlink(config_path)
                os.unlink(vocab_path)


if __name__ == '__main__':
    pytest.main([__file__]) 