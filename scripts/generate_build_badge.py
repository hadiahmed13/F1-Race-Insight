#!/usr/bin/env python3
"""Generate a build status badge for the F1 Race Insight project."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the badge output path
BUILD_BADGE_PATH = 'docs/images/build_status.png'

def generate_build_badge(status="passing"):
    """Generate a build status badge and save it to file.
    
    Args:
        status: The build status (passing, failing, etc.)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(2, 0.5))
    
    # Set status color
    if status.lower() == "passing":
        color = "green"
    elif status.lower() == "failing":
        color = "red"
    else:
        color = "yellow"
    
    # Create badge
    ax.add_patch(Rectangle((0, 0), 0.5, 1, facecolor='dimgray', edgecolor='none'))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 1, facecolor=color, edgecolor='none'))
    
    # Add text
    ax.text(0.25, 0.5, "build", ha='center', va='center', color='white', fontweight='bold')
    ax.text(0.75, 0.5, status, ha='center', va='center', color='white', fontweight='bold')
    
    # Format plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Remove whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(BUILD_BADGE_PATH), exist_ok=True)
    
    # Save badge
    plt.savefig(BUILD_BADGE_PATH, dpi=300, transparent=True)
    print(f"Build badge saved to {BUILD_BADGE_PATH}")

if __name__ == "__main__":
    generate_build_badge("passing") 