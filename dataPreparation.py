import os
import urllib
import urllib.request
from zipfile import ZipFile

URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fasion_mnist_images.zip"
FOLDER = "fasion_mnist_images"

if not os.path.isfile(FILE):
    print(f"DOwnloading {URL} and saving as {FILE}")
    urllib.request.urlretrieve(URL, FILE)

with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print("Done!")