import os
import zipfile


dataset_links = [
    "edomingo/catalonia-water-resource-daily-monitoring",
    "anikannal/solar-power-generation-data",
    "garystafford/environmental-sensor-data-132k",
    "loveall/appliances-energy-prediction",
]
dataset_folders = [link.split("/")[1] for link in dataset_links]

for dataset_link, dataset_folder in zip(dataset_links, dataset_folders):
    ## DOWNLOAD FROM KAGGLE
    os.system(f'kaggle datasets download -d "{dataset_link}"')
    print(f"downloading {dataset_link}")

    ## UNZIP
    path_to_zip_file = dataset_folder + ".zip"
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        zip_ref.extractall(dataset_folder)
