#!/bin/bash
cd /home/site/wwwroot/code
gunicorn --config gunicorn.conf.py startup:app