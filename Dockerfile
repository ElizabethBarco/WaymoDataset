FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install waymo-open-dataset-tf-2-12-0==1.6.7 numpy opencv-python

COPY load_dataset.py .


CMD ["python", "load_dataset.py"]