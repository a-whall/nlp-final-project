"""
Purpose:
    Automate refining our collection of AItA submissions into a dataset ready for training.
    Download the Dataset from google drive and unzip if necessary.

Usage:
    Run the following command from the project root directory:
    python3 scripts/build-dataset.py
"""
import os
import re
import csv
import zst
import html
import gdown
import json
from zipfile import ZipFile


DATAPATH = "./data/"

RAW_DATA_DIR = "AItAS"

URL = "https://drive.google.com/uc?id=1ui31UsnJbTLacMwWgfv81H7k6lG1atcu"


def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bEdit\b.*', '', text, flags=re.IGNORECASE)
    text = html.unescape(text)
    text = re.sub(r'&amp;amp;|&amp;', 'and', text, flags=re.IGNORECASE)
    return text

def valid_label(text):
    return text == "Not the A-hole" or text == "Asshole" or text == "No A-holes here" or text == "Everyone Sucks" or text == "Not enough info"

if __name__ == "__main__":

    os.makedirs(DATAPATH, exist_ok=True)

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

    out_file = open(DATAPATH + "AItAS_dataset.csv", "w", encoding="utf8")

    writer = csv.writer(out_file)

    obj_keys = ["title", "selftext", "link_flair_text"]

    writer.writerow(["title", "body", "label"])

    total_submissions = 0
    total_removed = 0
    
    for file_name in sorted(os.listdir(path_to_dir)):
        
        if not file_name.endswith(".zst"):
            continue

        print(f"Processing {file_name}...")

        path_to_file = os.path.join(path_to_dir, file_name)

        success = 0
        bad_lines = 0
        csv_bad_char = 0
        invalid_label = 0

        for line, bytes_processed in zst.read_lines(path_to_file):

            try:
                obj = json.loads(line)

                if not valid_label(obj["link_flair_text"]):
                    invalid_label += 1
                    continue

                writer.writerow(clean_text(obj[key]) for key in obj_keys)
            
            except KeyError as k:
                print(f"Invalid keys: {k}")
                exit()
            except json.JSONDecodeError as j:
                bad_lines += 1
            except UnicodeEncodeError as u:
                csv_bad_char += 1

            success += 1
        
        total_submissions += success
        removed = bad_lines + csv_bad_char + invalid_label
        total_removed += removed

        print(f"Extracted {success}/{success+removed} : {bad_lines} json decode errors : {csv_bad_char} invalid csv characters : {invalid_label} invalid labels")

    print(f"Finished. Final dataset has {total_submissions} total submissions. Removed {total_removed}")

    out_file.close()