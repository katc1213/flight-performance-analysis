import pandas as pd
import glob
import os

folder_path = r'C:\Users\Kat\Documents\repos\flight-performance-analysis\data'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

df_list = []

# Read each file into a dataframe and append to the list
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

merged_df.fillna(0)
merged_df['FL_DATE'] = df['FL_DATE'].str.replace('0:00', '')

merged_df.to_csv('combined_files.csv', index=False)
