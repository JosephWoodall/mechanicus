FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code - fix the path
COPY src/data_collection.py .

# Command to run the data collector service
CMD ["python", "data_collection.py"]