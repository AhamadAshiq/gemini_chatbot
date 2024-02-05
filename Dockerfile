FROM python:3.9

# Set the working directory in the container
WORKDIR /gemini

# Upgrade pip
RUN /usr/local/bin/python -m pip install --upgrade pip

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
