"""
Purpose:
    Extract r/AmItheAsshole submission lines from a zstandard-compressed NDJSON file.
    Write extracted lines to a new file called 'AItAS_YYYY-MM.NDJSON'.

Usage:
    Run the following command from the project root directory:
    python3 scripts/extract-aita-from-rs-zst.py path/to/RS_YYYY-MM.zst

Adapted from:
    https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/single_file.py
"""
import os
import sys
import zst
import json
from datetime import datetime


DATAPATH = "./data/"


if __name__ == "__main__":

    file_path = sys.argv[1]
    file_size = os.stat(file_path).st_size
    file_name = f"AItAS_{file_path[-11:-4]}.ndjson"

    good_lines = 0
    bad_lines = 0
    bytes_processed = 0
    created = None

    aita_posts_found = 0
    aita_deleted = 0
    aita_no_flair = 0
    aita_context_missing = 0

    batch_str = ""

    if not os.path.exists(DATAPATH):
        os.mkdir(DATAPATH)

    out_file = open(DATAPATH+file_name, 'w', encoding='utf8')

    for line, bytes_processed in zst.read_lines(file_path):

        try:
            obj = json.loads(line)

            created = datetime.utcfromtimestamp(int(obj['created_utc']))

            if obj["subreddit"] == "AmItheAsshole":
                aita_posts_found += 1
                if obj["title"] == "[deleted by user]":
                    aita_deleted += 1
                elif obj["link_flair_text"] == None:
                    aita_no_flair += 1
                elif obj["selftext"] == "[deleted]" or obj["selftext"] == "[removed]":
                    aita_context_missing += 1
                else:
                    batch_str += line + '\n'

        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1

        good_lines += 1

        if good_lines % 100000 == 0:
            print(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {good_lines:,} : {bad_lines:,} : {bytes_processed:,} : {(bytes_processed / file_size) * 100:.1f}% : ({aita_posts_found} found)")
            out_file.write(batch_str)
            batch_str = ""

    out_file.write(batch_str)
    out_file.close()

    total_removed = aita_no_flair + aita_deleted + aita_context_missing

    # Append statistics to file for later.
    stats = f"{good_lines:,} total posts.\nFound {aita_posts_found:,} AItA posts.\n{aita_no_flair:,} had no flair.\n{aita_deleted:,} were deleted by user.\n{aita_context_missing:,} had selftext removed or deleted.\nIn total {total_removed:,} AItA posts were ignored."
    with open(DATAPATH+"extraction-stats.txt", 'a') as stats_file:
        stats_file.write(f"{file_path}\n{stats}\n\n")

    print(f"Complete : {good_lines:,} : {bad_lines:,}\n{stats}")