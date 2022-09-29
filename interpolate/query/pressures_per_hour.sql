select datetime, group_concat(coalesce(air_pressure, ""))
from (
    select * 
    from air_pressure 
    order by datetime, amedas_station
)
where datetime between "2016" and "2020" 
group by datetime(datetime) 
order by datetime
;
