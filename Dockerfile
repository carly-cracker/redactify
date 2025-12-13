# Dockerfile

# 1. Use a Python image that includes necessary build tools
FROM python:3.9-slim

# 2. Set environment variables to avoid issues with Python and Streamlit
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 3. Create the working directory
WORKDIR /app

# 4. Copy and install dependencies
# We install the core requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch specifically for CPU to keep the image size manageable
# Note: You need to confirm the Python version (3.9 in this example) matches your local environment.
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# 5. Copy the rest of the application files
# This includes app.py, redactify_vid.py, etc.
COPY . .

# 6. Copy the YOLO models into the location your script expects
# The script looks for models one level up (../models). 
# We copy them into the container's root working directory /app/models.
# You MUST ensure your models are in the '../models' folder on your host machine before building!
COPY ../models /app/models 

# 7. Expose the default Streamlit port
EXPOSE 8501

# 8. Define the command to start the Streamlit application
CMD ["streamlit", "run", "app.py"]