# Use an official Python runtime as a parent image
# Using a slim image reduces the overall image size
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir flag is used to prevent caching package data, reducing image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py and the templates directory
COPY . .

# Expose the port the app runs on
# Flask and Gunicorn will be configured to listen on this port
EXPOSE 5000

# Command to run the application
# This is the default command, but it will be overridden by docker-compose
# based on the environment (development or production)
# In development, Flask's built-in server is used (flask run)
# In production, Gunicorn is used (gunicorn)
# The actual command is defined in docker-compose.yml
CMD ["python", "app.py"]
