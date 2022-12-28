/* Arrival Delayed Flights 

CREATE TABLE delayed_arr_flights AS
SELECT *
FROM flights
WHERE arr_delay>0 ;

/* Percentage of Delayed flights per airline and airport */

CREATE TABLE delays_airline_airport AS
SELECT a.origin, a.dest, b.name AS Name, b.carrier, SUM(a.dep_delay>0) AS No_of_Delayed_Flights, ROUND(SUM(a.dep_delay>0)*1.0 / COUNT(*), 2) AS Percent_of_dep_delayed_flights, COUNT(*) AS total_no_of_flights
FROM flights AS a, Airlines AS b
WHERE a.carrier=b.carrier
GROUP BY 1,2,3
ORDER BY 1,3 DESC;

/* Percentage of Departure Delayed flights per airport*/

CREATE TABLE flight_delays_per_airport AS
SELECT CASE WHEN p.engine in ('Turbo-fan','Turbo-jet') THEN 'Big Flights'
            ELSE 'Small Flights' END AS Engine, a.name AS Airports_Name, f.origin AS Airports, p.manufacturer, ROUND(SUM(f.dep_delay>0)*1.0 / COUNT(*),2) AS Percent_of_dep_delayed_flights, ROUND(AVG(f.dep_delay),0.01) AS Avg_dep_delays
FROM Airports AS a, flights AS f, planes AS p 
WHERE a.faa = f.origin AND
      f.tailnum = p.tailnum
GROUP BY 1,2,3
ORDER BY 1,2;

/* Percentage of Departure Delayed flights per airline*/

CREATE TABLE flight_delays_per_airline AS
SELECT CASE WHEN p.engine in ('Turbo-fan','Turbo-jet') THEN 'Big Flights'
            ELSE 'Small Flights' END AS Engine, a.name AS Airline, f.origin AS Airports, a.carrier, p.manufacturer, ROUND(SUM(f.dep_delay>0)*1.0 / COUNT(*),2) AS Percent_of_dep_delayed_flights, ROUND(AVG(f.dep_delay),0.01) AS Avg_dep_delays
FROM Airlines AS a, flights AS f, planes AS p 
WHERE a.carrier = f.carrier AND
      f.tailnum = p.tailnum
GROUP BY 1,2,3,4
ORDER BY 1,2;

/* Percentage of Departure Delayed flights per planes size and per airlines*/

CREATE TABLE flight_delays_per_airline_size AS
SELECT CASE WHEN p.engine in ('Turbo-fan','Turbo-jet') THEN 'Big Flights'
            ELSE 'Small Flights' END AS Engine, a.name AS Airline, a.carrier, p.manufacturer, ROUND(SUM(f.dep_delay>0)*1.0 / COUNT(*),2) AS Percent_of_dep_delayed_flights, COUNT(*) AS total_no_of_flights
FROM Airlines AS a, flights AS f, planes AS p 
WHERE a.carrier = f.carrier AND
      f.tailnum = p.tailnum 
GROUP BY 1,2,3
ORDER BY 1,3 DESC;

/* Percentage avg time Delayed flights per planes' size and per airlines*/

CREATE TABLE flight_delays_avg_per_airline_size AS
SELECT CASE WHEN p.engine in ('Turbo-fan','Turbo-jet') THEN 'Big Flights'
            ELSE 'Small Flights' END AS Engine, a.name AS Airline, a.carrier, p.manufacturer, ROUND(AVG(f.dep_delay) , 0.01) AS Avg_dep_delayed_flights
FROM Airlines AS a, flights AS f, planes AS p
WHERE a.carrier = f.carrier AND 
      p.tailnum = f.tailnum
GROUP BY 1,2,3
ORDER BY 1,4 DESC;

/* Total No of Delayed Flights based on Origin */

CREATE TABLE no_of_flights_origin AS
SELECT origin, SUM(dep_delay>0) AS no_of_delay_dep_flights, COUNT(origin) AS total_flights, ROUND(SUM(dep_delay>0)*1.0/COUNT(origin),2) AS Perc_of_delays_origin
FROM flights
GROUP BY 1
ORDER BY 3 DESC;

/* Total No of Delayed Flights per Airline */

CREATE TABLE no_of_flights_per_airline AS
SELECT a.name AS Airline, a.carrier, COUNT(b.flight) AS Total_no_of_flights, SUM(b.dep_delay>0 OR b.arr_delay>0) AS Delayed_flights, SUM(b.dep_delay<=0 AND b.arr_delay<=0) AS Ontime_flights
FROM Airlines AS a, flights AS b
WHERE a.carrier=b.carrier
GROUP BY 1,2
ORDER BY 1 DESC, 2 DESC, 3 DESC;

/* Percentage Delays per no of engines */

CREATE TABLE No_of_engines AS
SELECT (CASE WHEN (p.engines=1) THEN 'One Engine'
             WHEN (p.engines=2) THEN 'Two Engines'
             WHEN (p.engines=3) THEN 'Three Engines'
             ELSE 'Four Engines' END) AS No_of_engines, p.manufacturer, COUNT(f.flight) AS Total_flights, SUM(f.dep_delay>0 OR f.arr_delay>0) AS Delayed_flights, ROUND(SUM(f.dep_delay>0 OR f.arr_delay>0)*1.0 /COUNT(f.flight),2) AS Perc_delayed_flights
FROM flights AS f, planes AS p
WHERE f.tailnum = p.tailnum
GROUP BY 1
ORDER BY 5 DESC;

/* Percentage Delays per plane distance */

CREATE TABLE Airplane_Route_types AS
SELECT (CASE WHEN (f.distance<=1500) THEN 'Short-Haul Flights'
             WHEN (f.distance > 1500 AND f.distance <= 2200 ) THEN 'Medium-Haul Flights'
             ELSE 'Long-Haul Flights' END) AS Plane_types, COUNT(f.flight) AS Total_flights, SUM(f.dep_delay>0 OR f.arr_delay>0) AS Delayed_flights, ROUND(SUM(f.dep_delay>0 OR f.arr_delay>0)*1.0 /COUNT(f.flight),2) AS Perc_delayed_flights
FROM flights AS f
GROUP BY 1
ORDER BY 4 DESC;

/* Percentage Delays per no of Airplane seats */

CREATE TABLE No_of_seats AS
SELECT (CASE WHEN (p.seats<=50) THEN '2-50 Seats'
             WHEN (p.seats<=150) THEN '51-150 Seats'
             WHEN (p.seats<=250) THEN '151-250 Seats'
             WHEN (p.seats<=350) THEN '251-350 Seats'
             ELSE '351-450 Seats' END) AS no_of_seats, COUNT(f.flight) AS Total_flights, SUM(f.dep_delay>0 OR f.arr_delay>0) AS Delayed_flights, ROUND(SUM(f.dep_delay>0 OR f.arr_delay>0)*1.0 /COUNT(f.flight),2) AS Perc_delayed_flights
FROM flights AS f, planes AS p
WHERE f.tailnum = p.tailnum
GROUP BY 1
ORDER BY 4 DESC;

/* Arrival Flight Delays on All Airports */

CREATE TABLE Arrival_Delays_all_airports AS
SELECT a.name AS airport_name, a.alt, a.lon, SUM(f.arr_delay) AS Delay
FROM flights AS f, Airports AS a
GROUP BY 1,2;

/* Total Summary Delays */

CREATE TABLE Summary_delays AS
SELECT a.carrier, a.name, f.origin, f.month, f.day, f.hour, COUNT(f.flight) AS sum_of_flights, SUM(dep_delay) AS total_dep_delay, AVG(dep_delay) AS Avg_dep_delay
FROM flights AS f, Airlines AS a
WHERE a.carrier = f.carrier
GROUP BY 1,2,3,4,5,6;

/* Weather in flightsperhour */

CREATE TABLE weather_in_flights_per_hour AS
SELECT w.origin AS origin, w.month AS month, w.day AS day, w.hour AS hour, AVG(w.humid) AS humid, AVG(w.temp) AS temp, AVG(wind_speed) AS wind_temp, AVG(wind_gust) AS wind_gust, AVG(w.precip) AS precip,
AVG(w.pressure) AS pressure, AVG(w.visib) AS visib, AVG(f.dep_delay) AS dep_delay, AVG(f.air_time) AS air_time, AVG(f.distance) AS distance, a.lat AS latitude, a.lon AS longitude
FROM weather AS w, flights AS f, Airports AS a
WHERE w.hour=f.hour AND
      w.day=f.day AND
      w.month=f.month AND
      w.year=f.year AND
      w.origin=f.origin AND
      a.faa=f.origin
GROUP BY w.hour, w.day, w.month, w.year, w.origin; 

/* Flight delays per Month */ 

CREATE TABLE flight_delays_per_month AS
SELECT month AS Month, origin AS Origin, COUNT(flight) AS No_of_flights, ROUND(AVG(dep_delay),2) AS Average_Delays
FROM flights
GROUP BY 1;

/* Maximum Delay per Month */ 

CREATE TABLE Max_per_month AS
SELECT a.month, a.day, a.Avg_dep_delay
FROM (  SELECT Month, Day, ROUND(AVG(dep_delay),2) AS Avg_dep_delay
        FROM flights
        GROUP BY 1,2) as a
GROUP BY month
HAVING a.Avg_dep_delay = MAX(a.Avg_dep_delay);




