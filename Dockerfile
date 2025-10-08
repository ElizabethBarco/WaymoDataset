FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

RUN pip install waymo-open-dataset-tf-2-12-0==1.6.7

COPY load_dataset.py .

CMD ["python", "load_dataset.py"]