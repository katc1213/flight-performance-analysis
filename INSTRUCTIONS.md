
# INSTRUCTIONS

## Project Overview
This project analyzes historical U.S. flight performance data. The goal is to build a pipeline that stores flight data in a database and uses it for analysis and machine learning.

At the current stage of the project, we have:
- collected flight datasets from the Bureau of Transportation Statistics (BTS)
- written a Python script to merge monthly datasets
- preprocessed and cleaned merged flight datasets for analysis
- created a MySQL schema for storing the flight data
- created an ERD diagram for the database structure
- created a notebook with initial exploratory and descriptive visualizations

---

## Repository Files

`data/`  
Contains the raw flight datasets.

`clean_data/`  
Contains the cleaned flight datasets used for analysis. (does not currently contain merged flights dataset because file is too large to upload)

`merge_flights.py`  
Python script that merges the monthly flight datasets into one dataset.

`flight_db_create.sql`  
SQL script that creates the MySQL database schema.

`flight_db.drawio`  
Diagram showing the database structure and relationships between tables.

`analysis.ipynb`  
Notebook containing exploratory and descriptive analysis with visualizations.

`Checkpoint Reports/`  
Project checkpoint reports and documentation.

---

## How to Run the Current Workflow

### 1. Merge the datasets

Run:

```

python merge_flights.py

```

This combines the monthly flight datasets into a single dataset.

---

### 2. Create the database

Run the SQL script:

```

flight_db_create.sql

```

This creates the MySQL database and tables for storing the flight data.

---

### 3. Run the analysis notebook

Open the notebook:

```

analysis.ipynb

```

This contains the current exploratory and descriptive visualizations for the project.

---

## Data Source

Flight data was downloaded from the Bureau of Transportation Statistics:

https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr


