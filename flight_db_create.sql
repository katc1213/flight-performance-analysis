set global transaction isolation level serializable;
set global SQL_MODE = 'ANSI,TRADITIONAL';
SET SESSION sql_mode = 'REAL_AS_FLOAT,PIPES_AS_CONCAT,ANSI_QUOTES,IGNORE_SPACE,ONLY_FULL_GROUP_BY,ANSI,STRICT_TRANS_TABLES,STRICT_ALL_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,TRADITIONAL,NO_ENGINE_SUBSTITUTION';
set names utf8mb4;
set SQL_SAFE_UPDATES = 0;
set @thisDatabase = 'flight_db';

drop database if exists flight_db;
create database if not exists flight_db;
use flight_db;

create table airline (
  carrier_name char(10) not null,
  op_unique_carrier char(4) not null,
  primary key (carrier_name)
) engine = innodb;

create table airport (
  airport_id char(4) not null,
  city char(20) not null,
  state char(3) not null,
  primary key (airport_id)
) engine = innodb;

create table flight (
  -- OP_UNIQUE_CARRIER char(4) not null,
  -- OP_CARRIER_FL_NUM int not null,
  flight_id char(10) not null,
  flight_date date not null,
  depart_crs_time time not null,
  arrive_crs_time time not null,
  depart_time time not null,
  arrive_time time not null,
  origin char(4) not null,
  dest char(4) not null,
  carrier_delay int,
  weather_delay int,
  nas_delay int,
  security_delay int,
  primary key (flight_id)
) engine = innodb;


create table delay_type (
	delay_id int not null,
    delay_name char(20) not null
) engine = innodb;
