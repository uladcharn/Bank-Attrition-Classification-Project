import kagglehub

import os
import shutil

# Download latest version
path = kagglehub.dataset_download("sakshigoyal7/credit-card-customers")

print("Path to dataset files:", path)

# adding a dataset to the project file

destination = input("Please enter or copy your destination file here")

name = os.listdir(path)[0]

source_data_file = os.path.join(path, name)

shutil.move(source_data_file, destination)

print("The dataset {name} was moved to the project folder")


