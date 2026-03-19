import pandas as pd

# load data
# change filename if needed
df = pd.read_csv("/Applications/GitHub/flight-performance-analysis/data/AIRPORT_ORIGIN.csv")

# valid us states
states = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
    'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
    'VT','VA','WA','WV','WI','WY'
}

# filter
def is_us(desc):
    try:
        # split "City, ST: blah"
        state = desc.split(", ")[1][:2]
        return state in states
    except:
        return False

# apply filter
df_us = df[df["Description"].apply(is_us)]

# save
df_us.to_csv("airports_us_only.csv", index=False)

print("Done! Saved as airports_us_only.csv")
print("Original rows:", len(df))
print("US rows:", len(df_us))


# split Description column
df_us[["CityState", "AirportName"]] = df_us["Description"].str.split(": ", expand=True)
df_us[["City", "State"]] = df_us["CityState"].str.split(", ", expand=True)

# keep only needed columns
airport_df = df_us[["Code", "City", "State"]]

# rename to match SQL table
airport_df.columns = ["airport_id", "city", "state"]

# save
airport_df.to_csv("airports_us_only.csv", index=False)
