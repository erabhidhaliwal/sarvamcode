#!/usr/bin/env python3
"""
Sarvam-Flow: A local AI coding agent CLI powered by Sarvam-M.

Usage:
    python main.py run "Your request here"
    python main.py interactive
    python main.py scan
"""

from src.cli import app

if __name__ == "__main__":
    app()
