FROM tensorflow/tensorflow:2.4.1

WORKDIR /app

RUN pip install Pillow
RUN pip install SciPy
RUN pip install matplotlib