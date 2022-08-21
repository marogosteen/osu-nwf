from ml.datasets import wind_dataset
from ml.controllers.wind_controller import WindTrainController


for year in [2016, 2017, 2018, 2019]:
    print(year)
    train_dataset = wind_dataset.WindNWFDataset(
        generator=wind_dataset.DatasetGenerator(
            begin_year=2016,
            end_year=2020,
            target_year=year,
            forecast_timedelta=1,
            datasetname="train_pressureimage"
        ),
        datasetname="train_pressureimage"
    )
    eval_dataset = wind_dataset.WindNWFDataset(
        generator=wind_dataset.DatasetGenerator(
            begin_year=year,
            end_year=year+1,
            forecast_timedelta=1,
            datasetname="eval_pressureimage"
        ),
    datasetname="eval_pressureimage"
    )
    controller = WindTrainController(train_dataset, eval_dataset)
    controller.train()
