#!/bin/bash

# Navigate to project directory
cd /var/www/antigravity

# Pull latest changes
git pull origin main

# Stop existing Gunicorn process
pkill -f gunicorn

# Wait for process to stop
sleep 2

# Activate virtual environment
source venv/bin/activate

# Start Gunicorn
# Adjust workers and binding as needed for production
gunicorn --workers 3 --bind 127.0.0.1:5000 --timeout 120 run:app > logs/gunicorn.log 2>&1 &

echo "Deployment completed successfully!"
