# import config
# from ml.controllers.wave_controller import WaveTrainController

# # TODO
# # explain作る
# # グラフ化作る
# # クラス分類したい
# # obse, pred出力
# # Auto ML

# CONFIG_PATH = "config.json"

# train_config = config.NwfConfig(CONFIG_PATH)
# # for year in [2016, 2017, 2018, 2019]:
# for year in [2019]:
#     print(train_config.case_title+year)
#     train_config.eval_year = str(year)
#     # TODO __enter__ __exit__はdebugが辛いのでやめたい．
#     with WaveTrainController(train_config) as train_controller:
#         try:
#             train_controller.train()
#         # 必ずWaveTrainControllerの__exit__をcallして欲しい．
#         except KeyboardInterrupt as e:
#             print(e)
