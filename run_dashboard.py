#!/usr/bin/env python3
"""Wrapper script to run the F1 Race Insight dashboard with the correct Python path."""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now run the dashboard
os.system("streamlit run src/dashboard/Main.py") 