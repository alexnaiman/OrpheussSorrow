import json
import csv
import pandas
import os

"""
    Script used for merging Lakh Midi Dataset and Deezer Mood Emotion Dataset creating so our
    midi emotion dataset
"""


SCORE_FILE = '../../other_datasets/match_scores.json'
EMOTION_DATASET = '../../other_datasets/deezer_mood_detection_dataset-master/'
MY_DATASET_LOCATION = '../metadata/all_data.csv'
MATCHED_IDS = '../../other_datasets/matched_ids.txt'

with open(SCORE_FILE) as f:
    scores = json.load(f)

emotion_dataset_train = pandas.read_csv(
    EMOTION_DATASET + 'train.csv',
    usecols=[2, 3, 4])

emotion_dataset_test = pandas.read_csv(
    EMOTION_DATASET + 'test.csv',
    usecols=[2, 3, 4])
emotion_dataset_validation = pandas.read_csv(
    EMOTION_DATASET + 'validation.csv',
    usecols=[2, 3, 4])

emotion_dataset = pandas.concat(
    [emotion_dataset_train, emotion_dataset_test, emotion_dataset_validation])


partial_data_set = []
my_data_set = []
my_data_set2 = []
for i in emotion_dataset.values:
    partial_data_set.append(i[0])

for i in partial_data_set:
    if i in scores.keys():
        my_data_set.append(i)


with open(MATCHED_IDS) as f:
    matched_ids = set(list(map(lambda x: x.split()[1], f.readlines())))

for i in partial_data_set:
    if i in matched_ids:
        my_data_set2.append(i)

print("Data set size: %s" % len(my_data_set2))

finalDataset = emotion_dataset[emotion_dataset["MSD_track_id"].isin(
    my_data_set2)]

finalDataset.to_csv(MY_DATASET_LOCATION, index=False)
