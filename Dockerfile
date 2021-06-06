FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

WORKDIR /app

RUN pip install Pillow
RUN pip install SciPy