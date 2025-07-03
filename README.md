# AI Photo Tagger

An AI-powered photo tagging service that uses OpenCLIP to automatically tag your photos with relevant labels. This tool can process entire directories of images and generate tags based on a customizable vocabulary.

## Features

- **AI-Powered Tagging**: Uses OpenCLIP (ViT-B-32) model for accurate image recognition
- **Batch Processing**: Process entire directories of photos at once
- **Customizable Vocabulary**: Define your own set of tags in a simple text file
- **Confidence Thresholding**: Only include tags above a specified confidence level
- **Multiple Output Formats**: Save results as JSON or XMP metadata
- **GPU Acceleration**: Automatically uses CUDA if available, falls back to CPU

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ai-photo-tagger-lem
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

   Or if you prefer pip:

   ```bash
   pip install -e .
   ```

## Configuration

Create a `config.yaml` file in your project directory:

```yaml
photos_dir: ~/Pictures/inbox
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 0.2
output_format: xmp
```

### Configuration Options

- `photos_dir`: Directory containing photos to process (supports `~` for home directory)
- `model`: OpenCLIP model to use (currently supports `openclip://ViT-B-32`)
- `clip_vocab`: Path to vocabulary file with one tag per line
- `clip_top_k`: Number of top tags to consider per image
- `confidence_threshold`: Minimum confidence score (0.0 to 1.0) for tags to be included
- `output_format`: Output format (`json` or `xmp`)

## Vocabulary File

Create a `tags.txt` file with one tag per line. The system will use these tags to label your photos:

```
person
dog
cat
mountain
beach
car
food
# ... add more tags as needed
```

## Usage

### Command Line Interface

**Process all photos in configured directory**:

```bash
python -m ai_photo_tagger_lem.cli
```

**Process with custom config**:

```bash
python -m ai_photo_tagger_lem.cli --config my_config.yaml
```

**Process single image**:

```bash
python -m ai_photo_tagger_lem.cli --image photo.jpg
```

**Save as JSON instead of XMP**:

```bash
python -m ai_photo_tagger_lem.cli --output-format json
```

**Enable verbose logging**:

```bash
python -m ai_photo_tagger_lem.cli --verbose
```

### Python API

```python
from ai_photo_tagger_lem import PhotoTagger, Config

# Load configuration
config = Config('config.yaml')

# Create tagger
tagger = PhotoTagger(config)

# Process directory
results = tagger.process_directory()

# Save results
tagger.save_results(results, 'json')
```

## Output Formats

### JSON Output

Results are saved as `photo_tags.json`:

```json
{
  "/path/to/photo1.jpg": [
    ["person", 0.85],
    ["smiling", 0.72],
    ["portrait", 0.68]
  ],
  "/path/to/photo2.jpg": [
    ["mountain", 0.91],
    ["landscape", 0.78]
  ]
}
```

### XMP Output

XMP metadata is embedded directly into image files (requires additional libraries).

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## Performance

- **GPU**: Significantly faster processing with CUDA-enabled GPU
- **CPU**: Slower but functional processing on CPU
- **Memory**: Model requires ~1GB RAM for ViT-B-32

## Requirements

- Python 3.13+
- PyTorch
- OpenCLIP
- PIL (Pillow)
- PyYAML

## Troubleshooting

### Common Issues

1. **CUDA not available**: The system will automatically fall back to CPU processing
2. **Model download fails**: Check your internet connection and try again
3. **Memory errors**: Reduce batch size or use a smaller model
4. **No images found**: Check the `photos_dir` path in your config

### Debug Mode

Run with verbose logging to see detailed information:

```bash
python -m ai_photo_tagger_lem.cli --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- OpenCLIP for the vision-language model
- PyTorch for the deep learning framework
- The open-source community for inspiration and tools
