"""
Launcher for Project 2: Simple Motion Detection System
Runs the application.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import main

if __name__ == "__main__":
    main()
