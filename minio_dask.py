import minio
import sys
import os
import cv2
from sklearn import preprocessing

minio_key = 'Q3AM3UQ867SPQQA43P2F'
minio_secret = 'zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG'
bucket_name = 'san-2597'
minio_url = 'play.min.io:9000'
dask_endpoint_url = 'https://play.min.io:9000'
minioDataDir = './minioData/'


"""
The following section creates a Minio object and download data from the bucket 
if it exists in the Minio Server. 
"""

minioClient = minio.Minio(minio_url,access_key = minio_key, secret_key = minio_secret,secure=True)

if minioClient.bucket_exists(bucket_name):
	bucket_objs = minioClient.list_objects_v2(bucket_name,recursive=True)
else:
	sys.exit('Bucket not found, please check bucket name and re-run.')


#Reading files with dask

json_files = []
txt_files = []
csv_files = []

img_file_exts = ['.jpg','.jpeg','.bmp','.png','.tiff','.gif']

#Image files cannot be read remotely by dask, so have to be downloaded and then accessed
#Other files can be directly read from the remote server using dask 

for obj in bucket_objs:
	if os.path.splitext(obj.object_name)[1] in img_file_exts:
		pass
		# minioClient.fget_object(bucket_name,obj.object_name,minioDataDir+obj.object_name)
	elif os.path.splitext(obj.object_name)[1]=='.json':
		json_files.append('s3://'+bucket_name+'/'+obj.object_name)
	elif os.path.splitext(obj.object_name)[1]=='.txt':
		txt_files.append('s3://'+bucket_name+'/'+obj.object_name)
	elif os.path.splitext(obj.object_name)[1]=='.csv':
		csv_files.append('s3://'+bucket_name+'/'+obj.object_name)



""""""
"""
In this section image files are prepared with dask.array
Delayed loading of image files is done below
"""
""""""
import dask
import dask.array as da
path = minioDataDir+'DayNightData/'

"""
Label Encoding
"""
def convertLabels(label_list):
	num_labels = len(label_list)
	pre = preprocessing.LabelEncoder()
	label_list = pre.fit_transform(label_list)
	return label_list

"""
Read files from the directory
"""
def read_files(directory):
	features_list = list()
	label_list = list()
	num_classes = 0
	for root, dirs, files in os.walk(directory):
		for d in dirs:
			num_classes +=1
			images = os.listdir(root+d)
			for image in images:
				label_list.append(d)
				features_list.append(da.from_array(cv2.imread(root+d+image),chunks='auto'))
	label_list = convertLabels(label_list)
	return features_list,label_list



dataset = [dask.delayed(read_files)(path)]

images = [d[0] for d in dataset]
labels = [d[1] for d in dataset]


images = [da.from_delayed(im,shape=(100,458,800,3),dtype='uint8') for im in images] 
labels = [da.from_delayed(ia,shape=(100,2),dtype='int64') for ia in labels]

images = da.concatenate(images,axis=0)
labels = da.concatenate(labels,axis=0)

print("Images and labels have been loaded")


""""""
"""
Non-image files can be accessed directly from remote server using dask.dataframe function calls
"""
""""""
import dask.dataframe as dd
import dask.bag as db
import json
import time

minio_storage = {
	"key":minio_key, "secret":minio_secret,
	"client_kwargs": {
	"endpoint_url": dask_endpoint_url
	},
	"config_kwargs":{ "s3":{"addressing_style":"path"}}
}

print("Loading json data")
jsonData = dask.delayed(db.read_text)(json_files,storage_options=minio_storage).map(json.loads).to_dataframe()
# print(jsonData.compute().head())

print("Loading text data")
txtData = dask.delayed(db.read_text)(txt_files,storage_options=minio_storage).to_dataframe()
# print(txtData.compute().head(10,npartitions=2))

csvPath = '/media/san2597/New Volume/Downloads/train.csv'
parquetPath = '/media/san2597/New Volume/Downloads/train_csv.parquet'
print("Loading csv data")
csvData = dd.read_csv(csvPath)
# print(csvData.compute().head())


print("Saving csv data as parquet data")
# csvData.to_parquet('s3://'+bucket_name+'/train_csv.parquet',engine='fastparquet',compression='gzip',storage_options=minio_storage)

print("CSV file is saved in parquet format")

# csvData.to_parquet('./train_csv.parquet',engine='fastparquet',compression='gzip')

parquetData = dd.read_parquet(parquetPath,engine='fastparquet')


start = time.process_time()
print(parquetData.passenger_count.sum().compute())
print("Parquet data computation time: ",time.process_time()-start)

start = time.process_time()
print(csvData.passenger_count.sum().compute())
print("CSV data computation time: ",time.process_time()-start)

