"""
Command-line interface for AI Photo Tagger
"""

import argparse
import sys
import logging
from pathlib import Path

from .tagger import PhotoTagger
from .config import Config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI-powered photo tagging using OpenCLIP and Florence-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process photos using default config (OpenCLIP)
  python -m ai_photo_tagger_lem.cli
  
  # Process photos with Florence-2 (update config.yaml: model: florence2://base)
  python -m ai_photo_tagger_lem.cli
  
  # Process photos with custom config
  python -m ai_photo_tagger_lem.cli --config my_config.yaml
  
  # Process single image
  python -m ai_photo_tagger_lem.cli --image photo.jpg
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--image', '-i',
        help='Process single image instead of directory'
    )
    
    parser.add_argument(
        '--output-format', '-o',
        choices=['json', 'xmp'],
        help='Output format (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Create tagger
        tagger = PhotoTagger(config)
        
        if args.verbose:
            tagger.logger.setLevel(logging.DEBUG)
        
        if args.image:
            # Process single image
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Error: Image file not found: {args.image}")
                sys.exit(1)
            
            print(f"Processing single image: {image_path}")
            tags = tagger.tag_image(image_path)
            
            if tags:
                print(f"\nTags for {image_path.name}:")
                for tag, confidence in tags:
                    print(f"  {tag}: {confidence:.3f}")
            else:
                print("No tags found above confidence threshold")
        
        else:
            # Process directory
            print("Processing photos directory...")
            results = tagger.process_directory()
            
            if results:
                print(f"\nProcessed {len(results)} images")
                tagger.save_results(results, args.output_format)
            else:
                print("No images were successfully processed")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 