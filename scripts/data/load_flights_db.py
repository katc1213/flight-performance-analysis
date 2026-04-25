"""
load_flights_db.py
------------------
Flight Performance Analysis — CS 4365 Group 10
Loads merged_flights.csv + AIRPORT_ORIGIN.csv into flights.db (SQLite, 5 tables)

Run from repo root:
    python scripts/data/load_flights_db.py

Outputs:
    flights.db  — 5-table SQLite database ready for Tableau
"""

import pandas as pd
import sqlite3
import os

# Set working directory to repo root so all relative paths resolve correctly
import pathlib as _pl, os as _os
_os.chdir(_pl.Path(__file__).resolve().parent.parent.parent)

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
FLIGHTS_CSV  = 'data/CleanedData/merged_flights.csv'
AIRPORTS_CSV = 'data/AIRPORT_ORIGIN.csv'
DB_PATH      = 'data/CleanedData/flights.db'

# ─────────────────────────────────────────────
# 1. LOAD MAIN FLIGHT DATA
# ─────────────────────────────────────────────
print("Loading merged_flights.csv ...")
df = pd.read_csv(FLIGHTS_CSV, low_memory=False)
print(f"  -> {len(df):,} rows, {len(df.columns)} columns loaded")

# ─────────────────────────────────────────────
# 2. RENAME COLUMNS TO CLEAN SCHEMA NAMES
# ─────────────────────────────────────────────
df = df.rename(columns={
    'FL_DATE':             'flight_date',
    'FL_UNIQUE_NUM':       'fl_unique_num',
    'ORIGIN':              'origin',
    'DEST':                'dest',
    'CRS_DEP_TIME':        'crs_dep_time',
    'DEP_TIME':            'dep_time',
    'DEP_DELAY_NEW':       'dep_delay',
    'DEP_DEL15':           'dep_del15',
    'CRS_ARR_TIME':        'crs_arr_time',
    'ARR_TIME':            'arr_time',
    'ARR_DELAY_NEW':       'arr_delay',
    'ARR_DEL15':           'arr_del15',
    'CANCELLED':           'cancelled',
    'AIR_TIME':            'air_time',
    'FLIGHTS':             'flights',
    'DISTANCE':            'distance',
    'CARRIER_DELAY':       'carrier_delay',
    'WEATHER_DELAY':       'weather_delay',
    'NAS_DELAY':           'nas_delay',
    'SECURITY_DELAY':      'security_delay',
    'LATE_AIRCRAFT_DELAY': 'late_aircraft_delay',
    'TOTAL_ADD_GTIME':     'total_add_gtime',
})

# ─────────────────────────────────────────────
# 3. EXTRACT CARRIER + FLIGHT NUMBER FROM FL_UNIQUE_NUM
#    e.g. "AA1" -> carrier="AA", flight_num="1"
# ─────────────────────────────────────────────
df['carrier']    = df['fl_unique_num'].str.extract(r'^([A-Z]{2})')
df['flight_num'] = df['fl_unique_num'].str.extract(r'^[A-Z]{2}(\d+)')

print("\nCarrier extraction sample:")
print(df[['fl_unique_num', 'carrier', 'flight_num']].head(5).to_string(index=False))
print("Unique carriers found:", sorted(df['carrier'].dropna().unique().tolist()))

# ─────────────────────────────────────────────
# 4. BUILD flight_id
#    Format: AA_1_2025-01-01
# ─────────────────────────────────────────────
df['flight_id'] = (
    df['carrier'].astype(str) + '_' +
    df['flight_num'].astype(str) + '_' +
    df['flight_date'].astype(str)
)

# ─────────────────────────────────────────────
# 5. LOAD AIRPORT LOOKUP
#    Columns: Code, Description
#    Description format: "City, ST: Airport Name"
# ─────────────────────────────────────────────
print("\nLoading AIRPORT_ORIGIN.csv ...")
lookup = pd.read_csv(AIRPORTS_CSV)

lookup[['city_state', 'airport_name']] = (
    lookup['Description'].str.split(':', n=1, expand=True)
)
lookup['city']  = lookup['city_state'].str.rsplit(',', n=1).str[0].str.strip()
lookup['state'] = lookup['city_state'].str.rsplit(',', n=1).str[1].str.strip()
lookup = lookup.rename(columns={'Code': 'iata_code'})[
    ['iata_code', 'city', 'state', 'airport_name']
]
print(f"  -> {len(lookup):,} airport codes loaded")

# ─────────────────────────────────────────────
# 6. BUILD 5 TABLES
# ─────────────────────────────────────────────

# TABLE 1: airline
airlines = (
    df[['carrier']].drop_duplicates()
    .rename(columns={'carrier': 'carrier_code'})
    .sort_values('carrier_code')
    .reset_index(drop=True)
)

# TABLE 2: airport (enriched with lookup)
airports = pd.concat([
    df[['origin']].rename(columns={'origin': 'iata_code'}),
    df[['dest']].rename(columns={'dest': 'iata_code'})
]).drop_duplicates()
airports = airports.merge(lookup, on='iata_code', how='left')

unmatched = airports[airports['city'].isna()]
if len(unmatched) > 0:
    print(f"\n  WARNING: {len(unmatched)} airport codes not found in lookup:")
    print(" ", unmatched['iata_code'].tolist())
else:
    print(f"\n  All airport codes matched successfully")

# TABLE 3: flight (core flight info)
flights = df[[
    'flight_id', 'flight_date', 'carrier', 'flight_num', 'fl_unique_num',
    'origin', 'dest',
    'crs_dep_time', 'dep_time', 'dep_delay', 'dep_del15',
    'crs_arr_time', 'arr_time', 'arr_delay', 'arr_del15',
    'air_time', 'distance', 'flights'
]].copy()

# TABLE 4: delay (cause breakdown)
delays = df[[
    'flight_id',
    'carrier_delay', 'weather_delay', 'nas_delay',
    'security_delay', 'late_aircraft_delay', 'total_add_gtime'
]].copy()

# TABLE 5: cancellation (cancelled flights only)
cancellations = df[df['cancelled'] == 1][[
    'flight_id', 'cancelled'
]].copy()

# ─────────────────────────────────────────────
# 7. WRITE TO SQLITE
# ─────────────────────────────────────────────
print(f"\nWriting to {DB_PATH} ...")
conn = sqlite3.connect(DB_PATH)

airlines.to_sql(      'airline',      conn, if_exists='replace', index=False)
airports.to_sql(      'airport',      conn, if_exists='replace', index=False)
flights.to_sql(       'flight',       conn, if_exists='replace', index=False)
delays.to_sql(        'delay',        conn, if_exists='replace', index=False)
cancellations.to_sql( 'cancellation', conn, if_exists='replace', index=False)

conn.close()
print("Done.")

# ─────────────────────────────────────────────
# 8. SANITY CHECK
# ─────────────────────────────────────────────
print("\n-- Row counts --")
conn = sqlite3.connect(DB_PATH)
for table in ['airline', 'airport', 'flight', 'delay', 'cancellation']:
    n = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0]['n']
    print(f"  {table:<15} {n:>10,} rows")

print("\n-- Sample: flight table (first 3 rows) --")
print(pd.read_sql("SELECT * FROM flight LIMIT 3", conn).to_string(index=False))

print("\n-- Sample: airport table (first 5 rows) --")
print(pd.read_sql("SELECT * FROM airport LIMIT 5", conn).to_string(index=False))

conn.close()
print("\nflights.db is ready.")
print(f"  File size: {os.path.getsize(DB_PATH) / 1e6:.1f} MB")
print("  Connect Tableau: More > SQLite > select flights.db")
