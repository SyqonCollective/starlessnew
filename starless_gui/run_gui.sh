#!/bin/bash
# Launch script with OpenMP fix for macOS

export KMP_DUPLICATE_LIB_OK=TRUE
python gui_inference.py
