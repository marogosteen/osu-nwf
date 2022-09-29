select latitude, longitude
from(
    select distinct amedas_station 
    from air_pressure 
) as air_pressure
inner join amedas_station on amedas_station.id == air_pressure.amedas_station
order by id
;