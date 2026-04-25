import pandas as pd
import glob
import os
import pathlib
from datetime import time

# Set working directory to repo root so all relative paths resolve correctly
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)

# Raw monthly CSVs live in data/raw/
folder_path = 'data/raw'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

df_list = []

for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

# remove time from FL_DATE
merged_df['FL_DATE'] = pd.to_datetime(merged_df['FL_DATE'], format='mixed').dt.date
merged_df = merged_df.infer_objects().fillna(0)

merged_df[['OP_UNIQUE_CARRIER','OP_CARRIER_FL_NUM','ORIGIN','DEST']] = merged_df[['OP_UNIQUE_CARRIER','OP_CARRIER_FL_NUM','ORIGIN','DEST']].astype(str)

# make unique flight ID
merged_df['FL_CODE'] = merged_df['OP_UNIQUE_CARRIER'] + merged_df['OP_CARRIER_FL_NUM']

merged_df.set_index('FL_CODE', inplace=True)
merged_df.reset_index(inplace=True)

time_columns = ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']

def format_hhmm(val):
    val = int(val)
    if val == 2400:
        val = 0
    hours = val // 100
    minutes = val % 100
    # currently in 24 hr time
    # if hours > 12:
    #     hours -= 12
    return time(hour=hours, minute=minutes)

for col in time_columns:
    merged_df[col] = merged_df[col].apply(format_hhmm)

# no duplicated columns
#print(merged_df.duplicated().sum())

columns_to_drop = ['OP_UNIQUE_CARRIER','OP_CARRIER_FL_NUM','ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'DEST_CITY_NAME', 'DEST_STATE_ABR','DISTANCE_GROUP','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','FIRST_DEP_TIME','LONGEST_ADD_GTIME']
merged_df = merged_df.drop(columns=columns_to_drop)

print(merged_df.columns.tolist())
print(merged_df.shape)

merged_df.to_csv('data/CleanedData/merged_flights.csv', index=False)