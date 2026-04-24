import sqlite3, pandas as pd

# Set working directory to repo root so all relative paths resolve correctly
import pathlib as _pl, os as _os
_os.chdir(_pl.Path(__file__).resolve().parent.parent.parent)

conn = sqlite3.connect("data/CleanedData/flights.db")

for table in ["flight", "airline", "airport", "delay"]:
    print(f"\n--- {table} ---")
    print(pd.read_sql(f"PRAGMA table_info({table})", conn)[["name","type"]])

conn.close()