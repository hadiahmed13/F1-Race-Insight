#!/usr/bin/env python
"""Generate architecture diagram PNG from ASCII art."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from src.config import ARCHITECTURE_DIAGRAM

# Constants
FONT_SIZE = 14
LINE_HEIGHT = 18
IMAGE_PADDING = 20
OUTPUT_PATH = Path("docs/architecture_diagram.png")


def generate_diagram():
    """Generate architecture diagram as PNG from ASCII art."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    
    # Split diagram into lines and determine dimensions
    lines = ARCHITECTURE_DIAGRAM.strip().split("\n")
    max_line_length = max(len(line) for line in lines)
    
    # Create image with proper dimensions
    width = max_line_length * (FONT_SIZE * 0.6) + 2 * IMAGE_PADDING
    height = len(lines) * LINE_HEIGHT + 2 * IMAGE_PADDING
    
    # Create a white image
    image = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(image)
    
    # Try to load a monospace font, fall back to default if not available
    try:
        font = ImageFont.truetype("Courier", FONT_SIZE)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSansMono", FONT_SIZE)
        except IOError:
            font = ImageFont.load_default()
    
    # Draw each line
    for i, line in enumerate(lines):
        y_position = IMAGE_PADDING + i * LINE_HEIGHT
        draw.text((IMAGE_PADDING, y_position), line, font=font, fill="black")
    
    # Save the image
    image.save(OUTPUT_PATH)
    print(f"Architecture diagram saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_diagram() 