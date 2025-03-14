#!/bin/bash

# Install dependencies
python -m pip install -r requirements.txt

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Start the application
gunicorn --config gunicorn.conf.py startup:app