#!/usr/bin/env python3
"""
Setup script for AI Photo Tagger demo
"""

import os
import shutil
from pathlib import Path


def create_demo_directory():
    """Create a demo directory with sample images"""
    demo_dir = Path("demo_photos")
    demo_dir.mkdir(exist_ok=True)
    
    print(f"Created demo directory: {demo_dir}")
    print("Please add some test images to this directory.")
    print("Supported formats: jpg, jpeg, png, bmp, tiff, tif, webp")
    
    return demo_dir


def update_config_for_demo(demo_dir):
    """Update config.yaml to use demo directory"""
    config_content = f"""photos_dir: {demo_dir.absolute()}
model: openclip://ViT-B-32
clip_vocab: tags.txt
clip_top_k: 5
confidence_threshold: 0.2
output_format: json
"""
    
    with open('config.yaml', 'w') as f:
        f.write(config_content)
    
    print("Updated config.yaml to use demo directory")


def main():
    """Setup demo environment"""
    print("Setting up AI Photo Tagger demo environment...")
    
    # Check if config.yaml exists
    if not os.path.exists('config.yaml'):
        print("Creating config.yaml...")
        update_config_for_demo(create_demo_directory())
    else:
        print("config.yaml already exists")
    
    # Check if tags.txt exists
    if not os.path.exists('tags.txt'):
        print("Creating sample tags.txt...")
        sample_tags = """person
dog
cat
mountain
beach
car
food
tree
flower
building
"""
        with open('tags.txt', 'w') as f:
            f.write(sample_tags)
        print("Created sample tags.txt")
    else:
        print("tags.txt already exists")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Add some test images to the demo_photos directory")
    print("2. Run: python example.py")
    print("3. Or run: python -m ai_photo_tagger_lem.cli")
    print("\nFor more options, run: python -m ai_photo_tagger_lem.cli --help")


if __name__ == '__main__':
    main() 