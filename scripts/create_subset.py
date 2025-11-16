#!/usr/bin/env python3
"""
TACO Dataset Subset Creation Script

This script filters the TACO dataset to include only specified classes
and creates a new subset with filtered annotations and images.

Usage:
    python create_subset.py --class-ids 0 1 2 3 4
    python create_subset.py --class-ids 0 1 2 3 4 --data-dir ./data/TACO --output-dir ./data/taco_subset

Authors: Minahil Ali (22i-0849), Ayaan Khan (22i-0832)
Course: Deep Learning for Perception (CS4045)
Date: November 16, 2025
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create TACO dataset subset with specified classes'
    )
    parser.add_argument(
        '--class-ids',
        type=int,
        nargs='+',
        required=True,
        help='Class IDs to include in the subset (e.g., 0 1 2 3 4)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/TACO',
        help='Path to TACO dataset directory (default: ./data/TACO)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/taco_subset',
        help='Output directory for subset (default: ./data/taco_subset)'
    )
    parser.add_argument(
        '--annotation-file',
        type=str,
        default='annotations.json',
        help='Name of annotation file (default: annotations.json)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='images',
        help='Name of images directory relative to data-dir (default: images)'
    )
    
    return parser.parse_args()


def load_coco_annotations(ann_file):
    """Load COCO format annotations from JSON file."""
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"  Images: {len(coco_data.get('images', []))}")
    print(f"  Annotations: {len(coco_data.get('annotations', []))}")
    print(f"  Categories: {len(coco_data.get('categories', []))}")
    
    return coco_data


def filter_dataset(coco_data, class_ids):
    """Filter COCO dataset to include only specified classes."""
    print(f"\nFiltering dataset to classes: {class_ids}")
    
    # Filter annotations
    annotations = coco_data['annotations']
    filtered_annotations = [
        ann for ann in annotations 
        if ann['category_id'] in class_ids
    ]
    
    # Get image IDs that have at least one annotation from specified classes
    filtered_img_ids = set(ann['image_id'] for ann in filtered_annotations)
    
    # Filter images
    images = coco_data['images']
    filtered_images = [
        img for img in images 
        if img['id'] in filtered_img_ids
    ]
    
    # Filter categories
    categories = coco_data['categories']
    filtered_categories = [
        cat for cat in categories 
        if cat['id'] in class_ids
    ]
    
    print(f"\nFiltering results:")
    print(f"  Original - Images: {len(images)}, Annotations: {len(annotations)}")
    print(f"  Filtered - Images: {len(filtered_images)}, Annotations: {len(filtered_annotations)}")
    print(f"  Retention rate: {len(filtered_images)/len(images)*100:.1f}% images, "
          f"{len(filtered_annotations)/len(annotations)*100:.1f}% annotations")
    
    # Print per-class statistics
    class_counts = Counter(ann['category_id'] for ann in filtered_annotations)
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    print(f"\nPer-class statistics:")
    for cat_id in sorted(class_ids):
        cat_name = cat_id_to_name.get(cat_id, 'Unknown')
        count = class_counts.get(cat_id, 0)
        print(f"  ID {cat_id}: {cat_name:30s} - {count:5d} annotations")
    
    return filtered_images, filtered_annotations, filtered_categories


def create_subset_annotations(filtered_images, filtered_annotations, filtered_categories, coco_data):
    """Create new COCO annotation JSON with remapped IDs."""
    print("\nCreating subset annotations with remapped IDs...")
    
    # Get all filtered image IDs
    filtered_img_ids = set(img['id'] for img in filtered_images)
    
    # Re-map image and annotation IDs to be sequential starting from 1
    new_img_id_map = {
        old_id: new_id 
        for new_id, old_id in enumerate(sorted(filtered_img_ids), 1)
    }
    
    # Create new annotations with remapped IDs
    new_annotations = []
    for new_ann_id, ann in enumerate(filtered_annotations, 1):
        new_ann = ann.copy()
        new_ann['id'] = new_ann_id
        new_ann['image_id'] = new_img_id_map[ann['image_id']]
        new_annotations.append(new_ann)
    
    # Create new images with remapped IDs
    new_images = []
    for img in filtered_images:
        new_img = img.copy()
        new_img['id'] = new_img_id_map[img['id']]
        new_images.append(new_img)
    
    # Sort by ID for consistency
    new_images.sort(key=lambda x: x['id'])
    new_annotations.sort(key=lambda x: x['id'])
    
    # Build COCO JSON structure
    subset_coco_data = {
        'info': {
            'description': 'TACO Subset - Filtered Classes',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Minahil Ali (22i-0849), Ayaan Khan (22i-0832)',
            'date_created': '2025-11-16'
        },
        'licenses': coco_data.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations,
        'categories': filtered_categories
    }
    
    print(f"  Created {len(new_images)} image entries")
    print(f"  Created {len(new_annotations)} annotation entries")
    
    return subset_coco_data


def copy_images(filtered_images, src_img_dir, dst_img_dir):
    """Copy filtered images to subset directory."""
    print(f"\nCopying images from {src_img_dir} to {dst_img_dir}...")
    
    # Create destination directory
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    
    for img_info in tqdm(filtered_images, desc="Copying images"):
        src_path = src_img_dir / img_info['file_name']
        dst_path = dst_img_dir / img_info['file_name']
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"\nWarning: Source image not found: {src_path}")
            failed_count += 1
    
    print(f"\n  Successfully copied: {copied_count} images")
    if failed_count > 0:
        print(f"  Failed to copy: {failed_count} images")
    
    return copied_count, failed_count


def save_annotations(subset_coco_data, output_file):
    """Save subset annotations to JSON file."""
    print(f"\nSaving annotations to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(subset_coco_data, f, indent=2)
    
    print(f"  Annotations saved successfully")


def main():
    """Main function to create TACO subset."""
    args = parse_args()
    
    print("=" * 70)
    print("TACO Dataset Subset Creation Script")
    print("=" * 70)
    print(f"Class IDs to include: {args.class_ids}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ann_file = data_dir / args.annotation_file
    img_dir = data_dir / args.image_dir
    
    # Verify paths
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if not ann_file.exists():
        print(f"Error: Annotation file not found: {ann_file}")
        sys.exit(1)
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        sys.exit(1)
    
    # Load annotations
    coco_data = load_coco_annotations(ann_file)
    
    # Filter dataset
    filtered_images, filtered_annotations, filtered_categories = filter_dataset(
        coco_data, args.class_ids
    )
    
    # Create subset annotations
    subset_coco_data = create_subset_annotations(
        filtered_images, filtered_annotations, filtered_categories, coco_data
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    output_img_dir = output_dir / 'images'
    copied_count, failed_count = copy_images(
        filtered_images, img_dir, output_img_dir
    )
    
    # Save annotations
    output_ann_file = output_dir / 'annotations.json'
    save_annotations(subset_coco_data, output_ann_file)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUBSET CREATION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  - annotations.json: {output_ann_file}")
    print(f"  - images/: {output_img_dir} ({copied_count} images)")
    print("=" * 70)
    
    if failed_count > 0:
        print(f"\nWarning: {failed_count} images failed to copy")
        sys.exit(1)
    
    print("\n✓ Subset created successfully!")


if __name__ == '__main__':
    main()
