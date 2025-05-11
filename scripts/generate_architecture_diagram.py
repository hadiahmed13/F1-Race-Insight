#!/usr/bin/env python3
"""Generate an architecture diagram for the F1 Race Insight project."""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import networkx as nx

# Define the diagram output path
ARCHITECTURE_DIAGRAM = 'docs/images/architecture_diagram.png'

def generate_architecture_diagram():
    """Generate an architecture diagram and save it to file."""
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each component
    components = {
        "ETL Pipeline": "lightblue",
        "Feature Engineering": "lightgreen",
        "Model Training": "lightgreen",
        "API": "lightcoral",
        "Dashboard": "gold",
        "F1 Data Source": "lightgrey", 
        "Users": "lightgrey"
    }

    for component, color in components.items():
        G.add_node(component, color=color)

    # Add edges to show data flow
    edges = [
        ("F1 Data Source", "ETL Pipeline"),
        ("ETL Pipeline", "Feature Engineering"),
        ("Feature Engineering", "Model Training"),
        ("Model Training", "API"),
        ("API", "Dashboard"),
        ("Dashboard", "Users")
    ]

    G.add_edges_from(edges)

    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define node positions using a layered layout
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    
    # Draw nodes
    node_colors = [components[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color="grey", arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    
    # Add title
    plt.title("F1 Race Insight Architecture", fontsize=16, fontweight="bold")
    
    # Remove axis
    plt.axis("off")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(ARCHITECTURE_DIAGRAM), exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(ARCHITECTURE_DIAGRAM, dpi=300, bbox_inches="tight")
    
    print(f"Architecture diagram saved to {ARCHITECTURE_DIAGRAM}")

if __name__ == "__main__":
    generate_architecture_diagram() 