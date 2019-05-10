import settings

from train import Train_Runner


dataname = 'cora'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
task = 'link_prediction'         # 'clustering' or 'link_prediction'

settings = settings.get_settings(dataname, model, task)

runner = Train_Runner(settings)

runner.erun()