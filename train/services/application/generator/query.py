import config
from domain.tables import *
from services.application.generator import generate_mode


def generate_query(nwf_config: config.NwfConfig, mode: str) -> str:
    target_range_query = "NOT" if mode == generate_mode.train else ""

    select_colmuns = ",".join([
        UkbWindTable.time,
        UkbWindTable.velocity,
        UkbWindTable.sin_direction,
        UkbWindTable.cos_direction,
        KixWindTable.velocity,
        KixWindTable.sin_direction,
        KixWindTable.cos_direction,
        TomogashimaWindTable.velocity,
        TomogashimaWindTable.sin_direction,
        TomogashimaWindTable.cos_direction,
        AkashiWindTable.velocity,
        AkashiWindTable.sin_direction,
        AkashiWindTable.cos_direction,
        OsakaWindTable.velocity,
        OsakaWindTable.sin_direction,
        OsakaWindTable.cos_direction,
        TemperatureTable.temperature,
        AirPressureTable.air_pressure,
        WaveTable.significant_height,
        WaveTable.significant_period
    ])

    return f"""
    SELECT
        {select_colmuns}
    FROM
        Wind AS ukb
        INNER JOIN wind AS kix ON ukb.datetime == kix.datetime
        INNER JOIN wind AS tomogashima ON ukb.datetime == tomogashima.datetime
        INNER JOIN wind AS akashi ON ukb.datetime == akashi.datetime
        INNER JOIN wind AS osaka ON ukb.datetime == osaka.datetime
        INNER JOIN temperature ON ukb.datetime == Temperature.datetime
        INNER JOIN air_pressure AS air_pressure ON ukb.datetime == air_pressure.datetime
        INNER JOIN wave ON ukb.datetime == wave.datetime

    WHERE
        ukb.place == 'ukb' AND
        kix.place == 'kix' AND
        tomogashima.place == 'tomogashima' AND
        akashi.place == 'akashi' AND
        osaka.place == 'osaka' AND
        air_pressure.place == 'kobe' AND
        {target_range_query}(
            datetime(ukb.datetime) >= datetime("{nwf_config.eval_year}-01-01 00:00") AND
            datetime(ukb.datetime) <= datetime("{nwf_config.eval_year}-12-31 23:00")
        )

    ORDER BY
        ukb.datetime
    ;
    """
