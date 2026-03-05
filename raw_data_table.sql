USE flight_db;

CREATE TABLE raw_flights (

FL_DATE varchar(40),
OP_UNIQUE_CARRIER char(4),
OP_CARRIER_FL_NUM int,

ORIGIN char(4),
DEST char(4),

CRS_DEP_TIME int,
DEP_TIME int,
DEP_DEL15 float,

CRS_ARR_TIME int,
ARR_TIME int,
ARR_DEL15 float,

CANCELLED float,

CARRIER_DELAY float,
WEATHER_DELAY float,
NAS_DELAY float,
SECURITY_DELAY float,
LATE_AIRCRAFT_DELAY float

);