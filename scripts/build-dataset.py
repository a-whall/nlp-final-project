"""
Purpose:
    Automate refining our collection of AItA submissions into a dataset ready for training.
    Download the Dataset from google drive and unzip if necessary.

Usage:
    Run the following command from the project root directory:
    python3 scripts/build-dataset.py
"""
import os
import csv
import zst
import gdown
from zipfile import ZipFile


DATAPATH = "./data/"

RAW_DATA_DIR = "AItAS"

URL = "https://drive.google.com/uc?id=1uME2-98ErcED630U7kT8ceuotLDIuQj1"


if __name__ == "__main__":

    if not os.path.exists(DATAPATH):
        os.mkdir(DATAPATH)

    path_to_dir = DATAPATH + RAW_DATA_DIR
    path_to_zip = DATAPATH + RAW_DATA_DIR + ".zip"

    # Ensure data is ready

    if os.path.isdir(path_to_dir):
        pass

    elif os.path.isfile(path_to_zip):
        print("Unzipping...")
        with ZipFile(path_to_zip, 'r') as zip:
            zip.extractall(path_to_dir)

    else:
        user_yn = input("Download raw dataset? [y/n]: ")
        if user_yn.lower() == "y":
            gdown.download(URL, path_to_zip, quiet=False)
            print("Unzipping...")
            with ZipFile(path_to_zip, 'r') as zip:
                zip.extractall(path_to_dir)
        else:
            print(f"Move {RAW_DATA_DIR} or {RAW_DATA_DIR}.zip to {DATAPATH} and rerun the script.")
            exit()

    # Build the dataset

    out_file = open(DATAPATH + "AItAS_dataset.csv", "w")

    writer = csv.writer(out_file)

    writer.writerow(["title", "body", "label"])
    
    for file_name in sorted(os.listdir(path_to_dir)):
        
        if not file_name.endswith(".zst"):
            continue

        print(f"Processing {file_name}...")

        path_to_file = os.path.join(path_to_dir, file_name)

        for line, bytes_processed in zst.read_lines(path_to_file):

            pass # TODO: add the code for converting each line of json object to a line of csv

    out_file.close()