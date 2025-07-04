"""
XMP file handling for AI Photo Tagger
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
import exifread


class XMPHandler:
    """Handles XMP file operations and EXIF data integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_exiftool(self) -> bool:
        """Check if exiftool is available"""
        try:
            result = subprocess.run(['exiftool', '-ver'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_exif_data(self, image_path: Path) -> Dict[str, Any]:
        """Extract EXIF data from image using exifread"""
        exif_data = {}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
            # Only extract Make, Model, DateTimeOriginal using .printable
            wanted_tags = {
                'Image Make': 'Make',
                'Image Model': 'Model',
                'EXIF DateTimeOriginal': 'DateTimeOriginal',
            }
            for tag, clean_name in wanted_tags.items():
                if tag in tags:
                    value = tags[tag]
                    if hasattr(value, 'printable'):
                        clean_value = value.printable.strip()
                        if clean_value:
                            exif_data[clean_name] = clean_value
            self.logger.debug(f"Extracted {len(exif_data)} EXIF tags from {image_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to extract EXIF data from {image_path.name}: {e}")
        return exif_data
    
    def create_xmp_file(self, image_path: Path, tags: List[Tuple[str, float]], 
                       exif_data: Dict[str, Any]) -> Path:
        """Create a new XMP file with AI tags and EXIF data"""
        xmp_path = image_path.with_suffix('.xmp')
        
        # Register namespaces
        ET.register_namespace('x', 'adobe:ns:meta/')
        ET.register_namespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
        ET.register_namespace('exif', 'http://ns.adobe.com/exif/1.0/')
        ET.register_namespace('lr', 'http://ns.adobe.com/lightroom/1.0/')
        ET.register_namespace('AI', 'http://ai-photo-tagger.com/1.0/')
        
        # Create XMP XML structure
        xmp_root = ET.Element('{adobe:ns:meta/}xmpmeta', {
            '{adobe:ns:meta/}xmptk': 'AI Photo Tagger'
        })
        
        rdf = ET.SubElement(xmp_root, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
        
        # Description element
        desc = ET.SubElement(rdf, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description', {
            'xmlns:dc': 'http://purl.org/dc/elements/1.1/',
            'xmlns:exif': 'http://ns.adobe.com/exif/1.0/',
            'xmlns:lr': 'http://ns.adobe.com/lightroom/1.0/',
            'xmlns:AI': 'http://ai-photo-tagger.com/1.0/'
        })
        
        # Add AI-generated tags to dc:subject (standard keywords location)
        if tags:
            tag_list = ET.SubElement(desc, '{http://purl.org/dc/elements/1.1/}subject')
            bag = ET.SubElement(tag_list, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
            for tag, confidence in tags:
                li = ET.SubElement(bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                # Store only the tag name (singular, clean format)
                # Confidence stored separately or in custom namespace
                li.text = tag.capitalize()  # Ensure proper casing
                
            # Add confidence scores to custom namespace for reference
            if tags:
                confidence_list = ET.SubElement(desc, '{http://ai-photo-tagger.com/1.0/}ConfidenceScores')
                confidence_bag = ET.SubElement(confidence_list, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
                for tag, confidence in tags:
                    li = ET.SubElement(confidence_bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                    li.text = f"{tag}:{confidence:.3f}"
        
        # Add EXIF data
        if exif_data:
            for key, value in exif_data.items():
                # Map common EXIF tags to XMP
                if 'DateTime' in key:
                    ET.SubElement(desc, '{http://ns.adobe.com/exif/1.0/}DateTimeOriginal').text = value
                elif 'Make' in key:
                    ET.SubElement(desc, '{http://ns.adobe.com/exif/1.0/}Make').text = value
                elif 'Model' in key:
                    ET.SubElement(desc, '{http://ns.adobe.com/exif/1.0/}Model').text = value
                elif 'GPSLatitude' in key:
                    ET.SubElement(desc, '{http://ns.adobe.com/exif/1.0/}GPSLatitude').text = value
                elif 'GPSLongitude' in key:
                    ET.SubElement(desc, '{http://ns.adobe.com/exif/1.0/}GPSLongitude').text = value
        
        # Write XMP file
        tree = ET.ElementTree(xmp_root)
        ET.indent(tree, space="  ")
        tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Created XMP file: {xmp_path.name}")
        return xmp_path
    
    def update_xmp_file(self, xmp_path: Path, tags: List[Tuple[str, float]], 
                       exif_data: Dict[str, Any]) -> None:
        """Update existing XMP file with new AI tags"""
        try:
            # Parse existing XMP file with proper namespace handling
            ET.register_namespace('x', 'adobe:ns:meta/')
            ET.register_namespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
            ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
            ET.register_namespace('exif', 'http://ns.adobe.com/exif/1.0/')
            ET.register_namespace('lr', 'http://ns.adobe.com/lightroom/1.0/')
            ET.register_namespace('AI', 'http://ai-photo-tagger.com/1.0/')
            
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            # Find or create description element
            rdf = root.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF')
            if rdf is None:
                self.logger.warning(f"Could not parse existing XMP file: {xmp_path.name}")
                return
            
            desc = rdf.find('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description')
            if desc is None:
                desc = ET.SubElement(rdf, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description', {
                    'xmlns:dc': 'http://purl.org/dc/elements/1.1/',
                    'xmlns:exif': 'http://ns.adobe.com/exif/1.0/',
                    'xmlns:lr': 'http://ns.adobe.com/lightroom/1.0/',
                    'xmlns:AI': 'http://ai-photo-tagger.com/1.0/'
                })
            
            # Add new AI tags to dc:subject (standard keywords location)
            if tags:
                # Get existing keywords to avoid duplicates
                existing_subject = desc.find('.//{http://purl.org/dc/elements/1.1/}subject')
                existing_tags = set()
                if existing_subject is not None:
                    for li in existing_subject.findall('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li'):
                        if li.text:
                            existing_tags.add(li.text)
                
                # Remove existing subject to rebuild it
                if existing_subject is not None:
                    desc.remove(existing_subject)
                
                # Create new subject with all tags (existing + new)
                tag_list = ET.SubElement(desc, '{http://purl.org/dc/elements/1.1/}subject')
                bag = ET.SubElement(tag_list, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
                
                # Add existing tags back
                for tag in sorted(existing_tags):
                    li = ET.SubElement(bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                    li.text = tag
                
                # Add new AI tags (avoiding duplicates)
                for tag, confidence in tags:
                    clean_tag = tag.capitalize()
                    if clean_tag not in existing_tags:
                        li = ET.SubElement(bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                        li.text = clean_tag
                
                # Update confidence scores
                confidence_list = desc.find('.//{http://ai-photo-tagger.com/1.0/}ConfidenceScores')
                if confidence_list is not None:
                    desc.remove(confidence_list)
                
                confidence_list = ET.SubElement(desc, '{http://ai-photo-tagger.com/1.0/}ConfidenceScores')
                confidence_bag = ET.SubElement(confidence_list, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag')
                for tag, confidence in tags:
                    li = ET.SubElement(confidence_bag, '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li')
                    li.text = f"{tag}:{confidence:.3f}"
            
            # Write updated XMP file
            ET.indent(tree, space="  ")
            tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Updated XMP file: {xmp_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update XMP file {xmp_path.name}: {e}")
    
    def export_jpeg_with_xmp(self, image_path: Path, xmp_path: Path, 
                           output_dir: Path, max_size_mb: float = 1.0) -> Optional[Path]:
        """Export image as JPEG with embedded XMP data"""
        # Create output filename
        output_filename = image_path.stem + '.jpg'
        output_path = output_dir / output_filename
        
        try:
            # Use Pillow for JPEG conversion
            from PIL import Image
            
            # Open and convert image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Start with high quality and reduce if needed
                for quality in [95, 90, 85, 80, 75]:
                    img.save(output_path, 'JPEG', quality=quality, optimize=True)
                    
                    # Check file size
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    if file_size_mb <= max_size_mb:
                        self.logger.info(f"JPEG export successful (quality {quality}, {file_size_mb:.2f}MB)")
                        break
                else:
                    self.logger.warning(f"JPEG export too large ({file_size_mb:.2f}MB)")
            
            # Now embed XMP data if exiftool is available
            if self.check_exiftool() and xmp_path.exists():
                xmp_cmd = [
                    'exiftool',
                    '-overwrite_original',
                    f'-tagsfromfile={str(xmp_path)}',
                    '-all:all',
                    str(output_path)
                ]
                
                xmp_result = subprocess.run(xmp_cmd, capture_output=True, text=True, timeout=60)
                if xmp_result.returncode != 0:
                    self.logger.warning(f"Failed to embed XMP data: {xmp_result.stderr}")
                else:
                    self.logger.info(f"XMP data embedded successfully")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export WebP for {image_path.name}: {e}")
            return None
    
    def _fallback_webp_export(self, image_path: Path, output_dir: Path, max_size_mb: float) -> Optional[Path]:
        """Fallback WebP export using Pillow when exiftool fails"""
        try:
            from PIL import Image
            
            output_filename = image_path.stem + '.webp'
            output_path = output_dir / output_filename
            
            # Open and convert image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Start with high quality and reduce if needed
                for quality in [85, 75, 65, 55, 45]:
                    img.save(output_path, 'WEBP', quality=quality, optimize=True)
                    
                    # Check file size
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    if file_size_mb <= max_size_mb:
                        self.logger.info(f"Fallback WebP export successful (quality {quality}, {file_size_mb:.2f}MB)")
                        return output_path
                
                self.logger.warning(f"Fallback WebP export too large ({file_size_mb:.2f}MB)")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Fallback WebP export failed for {image_path.name}: {e}")
            return None
    
    def log_xmp_data(self, xmp_path: Path) -> None:
        """Log the XMP data for debugging"""
        try:
            # Register namespaces for parsing
            ET.register_namespace('x', 'adobe:ns:meta/')
            ET.register_namespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
            ET.register_namespace('dc', 'http://purl.org/dc/elements/1.1/')
            ET.register_namespace('exif', 'http://ns.adobe.com/exif/1.0/')
            ET.register_namespace('AI', 'http://ai-photo-tagger.com/1.0/')
            
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            # Extract and log dc:subject tags (keywords)
            keywords = []
            for li in root.findall('.//{http://purl.org/dc/elements/1.1/}subject//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li'):
                if li.text:
                    keywords.append(li.text)
            
            # Extract and log confidence scores
            confidence_scores = []
            for li in root.findall('.//{http://ai-photo-tagger.com/1.0/}ConfidenceScores//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li'):
                if li.text:
                    confidence_scores.append(li.text)
            
            # Extract and log EXIF data
            exif_data = {}
            for elem in root.findall('.//{http://ns.adobe.com/exif/1.0/}*'):
                tag = elem.tag.split('}')[-1]  # Remove namespace
                exif_data[tag] = elem.text
            
            self.logger.info(f"XMP data for {xmp_path.name}:")
            if keywords:
                self.logger.info(f"  Keywords: {', '.join(keywords)}")
            if confidence_scores:
                self.logger.info(f"  AI Confidence: {', '.join(confidence_scores)}")
            if exif_data:
                self.logger.info(f"  EXIF: {exif_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to log XMP data for {xmp_path.name}: {e}")
    
    def add_hierarchical_keywords(self, xmp_path: Path, hierarchical_tags: List[str]) -> None:
        """Add hierarchical keywords to lr:hierarchicalSubject"""
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            rdf = root.find('.//rdf:RDF', namespaces={'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
            if rdf is None:
                self.logger.warning(f"Could not parse XMP file: {xmp_path.name}")
                return
            
            desc = rdf.find('.//rdf:Description')
            if desc is None:
                desc = ET.SubElement(rdf, 'rdf:Description', {
                    'xmlns:lr': 'http://ns.adobe.com/lightroom/1.0/'
                })
            
            # Add hierarchical keywords
            if hierarchical_tags:
                hierarchical_list = ET.SubElement(desc, 'lr:hierarchicalSubject')
                hierarchical_bag = ET.SubElement(hierarchical_list, 'rdf:Bag')
                for tag in hierarchical_tags:
                    li = ET.SubElement(hierarchical_bag, 'rdf:li')
                    li.text = tag
            
            # Write updated XMP file
            ET.indent(tree, space="  ")
            tree.write(xmp_path, encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Added hierarchical keywords to {xmp_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add hierarchical keywords to {xmp_path.name}: {e}") 