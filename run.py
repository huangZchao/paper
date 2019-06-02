import settings

from train import Train_Runner


dataname = 'ca-cit-HepTh'

settings = settings.get_settings(dataname, 13)  # todo seq_len-2

runner = Train_Runner(settings)

runner.erun()