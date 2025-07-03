#!/usr/bin/env python3
"""
Example usage of AI Photo Tagger
"""

import os
from pathlib import Path
from ai_photo_tagger_lem import PhotoTagger, Config


def main():
    """Example of using the AI Photo Tagger"""
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found. Please create it first.")
        print("See README.md for configuration details.")
        return
    
    # Check if vocabulary file exists
    if not os.path.exists('tags.txt'):
        print("Error: tags.txt not found. Please create it first.")
        print("See README.md for vocabulary file format.")
        return
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = Config('config.yaml')
        
        # Create tagger
        print("Initializing AI Photo Tagger...")
        tagger = PhotoTagger(config)
        
        # Process photos directory
        print(f"Processing photos in: {config['photos_dir']}")
        results = tagger.process_directory()
        
        if results:
            print(f"\nSuccessfully processed {len(results)} images!")
            print("\nSample results:")
            
            # Show first few results
            for i, (image_path, tags) in enumerate(results.items()):
                if i >= 3:  # Show only first 3
                    break
                image_name = Path(image_path).name
                tag_names = [tag[0] for tag in tags]
                print(f"  {image_name}: {', '.join(tag_names)}")
            
            # Save results
            print(f"\nSaving results in {config.get('output_format', 'json')} format...")
            tagger.save_results(results)
            print("Done!")
        
        else:
            print("No images were successfully processed.")
            print("Check that:")
            print("  1. The photos directory exists and contains images")
            print("  2. The images are in supported formats (jpg, png, etc.)")
            print("  3. You have sufficient permissions to read the files")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("This might be due to:")
        print("  1. Missing dependencies (run: pip install -e .)")
        print("  2. Network issues (model download failed)")
        print("  3. Insufficient memory")


if __name__ == '__main__':
    main() 