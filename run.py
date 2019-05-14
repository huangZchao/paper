import settings

from train import Train_Runner


dataname = 'barbell'       # 'cora' or 'citeseer' or 'pubmed'

settings = settings.get_settings(dataname)

runner = Train_Runner(settings)

runner.erun()