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
  airport_id varchar(4) not null,
  city varchar(25) not null,
  state varchar(3) not null,
  primary key (airport_id)
) engine = innodb;

create table raw_origin (
  airport_id varchar(4) not null,
  origin_info varchar(128) not null,
  primary key (airport_id)
) engine = innodb;

create table flight (
  flight_id varchar(10) not null,
  flight_date date not null,
  
  origin_id varchar(4) not null,
  dest_id varchar(4) not null,
  
  crs_dep_time time not null,
  dep_time time,
  dep_total_delay int,
  dep_del15 boolean not null default 0,
  
  crs_arr_time time not null,
  arr_time time,
  arr_total_delay int,
  arr_del15 boolean not null default 0,
  
  cancelled boolean not null default 0,
  air_time int,
  flights tinyint,
  distance int,
  
  carrier_delay int,
  weather_delay int,
  nas_delay int,
  sec_delay int,
  late_aircraft_delay int,
  total_add_gtime int,
  
  primary key (flight_id, flight_date,crs_dep_time)
   -- constraint fk_name1 foreign key (origin_id) references airport (airport_id) on delete cascade on update cascade,
--    constraint fk_name2 foreign key (dest_id) references airport (airport_id) on delete cascade on update cascade
-- update later in separate file
) engine = innodb;


create table delay (
    flight_id varchar(10),
    
    carrier_delay int,
    weather_delay int,
    nas_delay int,
    security_delay int,
    late_aircraft_delay int,
    
    primary key (flight_id),
    foreign key (flight_id) references flight(flight_id) on delete cascade on update cascade
) engine = innodb;
