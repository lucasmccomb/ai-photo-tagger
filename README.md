# AI Photo Tagger

An AI-powered photo tagging service that uses OpenCLIP and Florence-2 to automatically tag your photos with relevant labels. This tool can process entire directories of images and generate tags using either a predefined vocabulary (OpenCLIP) or zero-shot generation (Florence-2).

## Features

- **AI-Powered Tagging**: Uses OpenCLIP (ViT-B-32) or Florence-2 for accurate image recognition
- **Multiple Models**: Choose between OpenCLIP (vocabulary-based) or Florence-2 (zero-shot generation)
- **Batch Processing**: Process entire directories of photos recursively
- **Customizable Vocabulary**: Define your own set of tags in a simple text file (OpenCLIP)
- **Zero-Shot Generation**: Generate tags without predefined vocabulary (Florence-2)
- **Confidence Thresholding**: Only include tags above a specified confidence level
- **XMP Metadata Integration**: Creates or updates XMP sidecar files with AI tags
- **EXIF Data Preservation**: Extracts and preserves existing EXIF data
- **JPEG Export**: Creates optimized JPEG versions with embedded metadata (max 1MB)
- **Non-Destructive**: Original images remain untouched
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
model: openclip://ViT-B-32 # or florence2://base for Florence-2
clip_vocab: tags.txt # only used for OpenCLIP
clip_top_k: 5
confidence_threshold: 0.6
output_format: xmp

# Florence-2 specific settings
florence_prompt: "List key nouns in this image, comma separated."
```

### Configuration Options

- `photos_dir`: Directory containing photos to process (supports `~` for home directory)
- `model`: Model to use:
  - `openclip://ViT-B-32` for OpenCLIP (vocabulary-based tagging)
  - `florence2://base` for Florence-2-base (zero-shot generation)
  - `florence2://large` for Florence-2-large (zero-shot generation)
- `clip_vocab`: Path to vocabulary file with one tag per line (only used for OpenCLIP)
- `clip_top_k`: Number of top tags to consider per image
- `confidence_threshold`: Minimum confidence score (0.0 to 1.0) for tags to be included
- `output_format`: Output format (`json` or `xmp`)
- `florence_prompt`: Custom prompt for Florence-2 tag generation

## Vocabulary File (OpenCLIP Only)

For OpenCLIP model, create a `tags.txt` file with one tag per line. The system will use these tags to label your photos:

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

**Note**: Florence-2 generates tags automatically without requiring a vocabulary file.

## Usage

### Command Line Interface

**Process all photos in configured directory (OpenCLIP)**:

```bash
python -m ai_photo_tagger_lem.cli
```

**Process with Florence-2** (update `config.yaml` first):

```bash
# Edit config.yaml: model: florence2://base
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

## Model Comparison

### OpenCLIP vs Florence-2

| Feature                 | OpenCLIP                        | Florence-2                        |
| ----------------------- | ------------------------------- | --------------------------------- |
| **Tagging Method**      | Vocabulary-based classification | Zero-shot generation              |
| **Vocabulary Required** | Yes (custom tags.txt file)      | No (generates tags automatically) |
| **Speed**               | Fast (pre-computed embeddings)  | Slower (generation per image)     |
| **Accuracy**            | Good for predefined concepts    | Excellent for diverse content     |
| **Memory Usage**        | Lower                           | Higher                            |
| **GPU Requirements**    | Moderate                        | High (recommended)                |

### When to Use Each Model

**Use OpenCLIP when:**

- You have a specific set of tags you want to detect
- You need fast processing of many images
- You're working with limited computational resources
- You want consistent, controlled vocabulary

**Use Florence-2 when:**

- You want to discover new tags automatically
- You have diverse, unpredictable image content
- You have powerful GPU resources available
- You want more natural, descriptive tags

## Workflow

The AI Photo Tagger now follows this workflow:

1. **Scan Directory**: Recursively finds all image files in the configured directory
2. **Generate AI Tags**: Uses OpenCLIP to analyze each image and generate relevant tags
3. **Extract EXIF Data**: Preserves existing EXIF metadata from original images
4. **Create/Update XMP Files**:
   - Creates new `.xmp` sidecar files if none exist
   - Updates existing `.xmp` files with new AI tags
   - Preserves existing XMP data and adds EXIF information
5. **Export JPEG**: Creates optimized JPEG versions with embedded metadata
6. **Log Results**: Shows detailed information about each processed image

### Output Structure

```
photos_dir/
├── original_image1.jpg
├── original_image1.xmp          # Created/updated with AI tags
├── subfolder/
│   ├── original_image2.png
│   └── original_image2.xmp      # Created/updated with AI tags
└── tagged-jpeg-exports/
    ├── original_image1.jpg      # Optimized JPEG with embedded metadata
    └── original_image2.jpg      # Optimized JPEG with embedded metadata
```

### XMP File Format

XMP files follow industry best practices with keywords stored in `dc:subject`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="AI Photo Tagger">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/"
                     xmlns:exif="http://ns.adobe.com/exif/1.0/"
                     xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
                     xmlns:AI="http://ai-photo-tagger.com/1.0/">
      <!-- Standard keywords - readable by all DAM software -->
      <dc:subject>
        <rdf:Bag>
          <rdf:li>Person</rdf:li>
          <rdf:li>Smiling</rdf:li>
          <rdf:li>Portrait</rdf:li>
        </rdf:Bag>
      </dc:subject>

      <!-- AI confidence scores (custom namespace) -->
      <AI:ConfidenceScores>
        <rdf:Bag>
          <rdf:li>Person:0.850</rdf:li>
          <rdf:li>Smiling:0.720</rdf:li>
          <rdf:li>Portrait:0.680</rdf:li>
        </rdf:Bag>
      </AI:ConfidenceScores>

      <!-- EXIF data -->
      <exif:DateTimeOriginal>2023:01:15 14:30:25</exif:DateTimeOriginal>
      <exif:Make>Canon</exif:Make>
      <exif:Model>EOS R5</exif:Model>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

### Keyword Best Practices

- **Standard Location**: Keywords stored in `dc:subject` (Dublin Core)
- **Format**: Singular nouns with proper casing (`Person`, not `people`)
- **No Duplicates**: System automatically avoids duplicate keywords
- **Confidence Scores**: Stored separately in custom namespace
- **EXIF Preservation**: All original EXIF data preserved
- **Hierarchical Support**: Can add `lr:hierarchicalSubject` for nested keywords

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
- exiftool (system dependency)
- exifread
- lxml

## Troubleshooting

### Common Issues

1. **CUDA not available**: The system will automatically fall back to CPU processing
2. **Model download fails**: Check your internet connection and try again
3. **Memory errors**: Reduce batch size or use a smaller model
4. **No images found**: Check the `photos_dir` path in your config
5. **exiftool not found**: Install exiftool on your system:
   - **macOS**: `brew install exiftool`
   - **Ubuntu/Debian**: `sudo apt-get install exiftool`
   - **Windows**: Download from https://exiftool.org/

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
