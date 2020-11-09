# Minio-access-with-Dask

## KEY IDEA
Access objects in a bucket on a MinIO storage server using Dask for Python.

## HOW?
Dask can read non-image files (.json, .csv, .text) stored remotely using the storage_options argument available for all Dask functions. To work with images stored remotely, they have to be first downloaded to the local system before being read by Dask (**Error 403** is reported by dask_image.imread.imread otherwise). The images are downloaded to a **"minioData"** folder created in the folder where the file is executed. 
The storage_options argument is modified to allow access to the public MinIO server (https://play.min.io), using a custom bucket created for this repository. The bucket contains the above mentioned file types as well as folders containing image files (.jpg).

## EXECUTION
1. Create a virtual environment with **Python 3.7** and install the required libraries using _pip install -r requirements.txt_
2. Modify initialization parameters in the python file as needed.
3. Run using _python minio_dask.py_


## TO DO
Use the given data with Dask to train ML models (CNN classifier for Image, LSTM-based classifier for text data) using TensorFlow.

