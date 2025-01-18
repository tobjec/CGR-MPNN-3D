# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary project files into the container
COPY activation_energy_predictor.py ./ 
COPY cgr_mpnn_3D ./cgr_mpnn_3D
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Define a volume for external data files
VOLUME ["/files"]

# Define the default command for the container
ENTRYPOINT ["python3", "activation_energy_predictor.py"]