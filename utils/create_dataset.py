# create dataset
import pandas as pd
import os
import json
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dir', type=str)

args = parser.parse_args()

# create json file
json_aesthetic_data = []
json_technical_data = []
# get all the folders
folders = [f for f in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, f))]

def create_label(value, technical=False):
    x = [2 for f in range(10)]
    x[int(value) - 1] = 30
    # if technical:
    #     x = [2 for f in range(10)]
    #     x[int(value) - 1] = 30
    # else:
    #     x = [0.01 for f in range(10)]
    #     x[int(value) - 1] = 0.91
    return x

# for each folder
for folder in folders:

    print(folder)
    images_folder_path = os.path.join(args.dir, folder)
    csv = pd.read_csv(os.path.join(args.dir, folder + '.csv'), header=0, dtype=str)

    # RENAME ALL THE ITEMS
    # for f in os.listdir(images_folder_path):
    #     print(f)
    #     os.rename(os.path.join(images_folder_path, f), os.path.join(images_folder_path, folder + "_" + f))
    # for i in range(len(csv)):
    #     # print(csv.iloc[0])
    #     csv.iloc[i][0] = folder + "_" + str(csv.iloc[i][0])
    # # print(csv)
    # csv.to_csv(os.path.join(args.dir, folder + '.csv'))

    # APPEND JSON DATA
    for i in range(len(csv)):
        # print(csv.iloc[i][0])
        json_technical_data.append({
            "image_id" : csv.iloc[i][1],
            "label" : create_label(csv.iloc[i][2], True) # technical
        })
        json_aesthetic_data.append({
            "image_id" : csv.iloc[i][1],
            "label" : create_label(csv.iloc[i][3], False) # aesthetic
        })

with open(os.path.join(args.dir, "samples_aesthetic.json"), "w") as f:
    json.dump(json_aesthetic_data, f)

with open(os.path.join(args.dir, "samples_technical.json"), "w") as f:
    json.dump(json_technical_data, f)

