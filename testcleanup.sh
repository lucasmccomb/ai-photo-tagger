#!/bin/bash

# Delete all XMP files recursively
find ~/Pictures/test-tagging -type f -name '*.xmp' -delete

# Delete all tagged-jpeg-exports directories recursively
find ~/Pictures/test-tagging -type d -name 'tagged-jpeg-exports' -exec rm -rf {} +

echo "Cleanup complete." 