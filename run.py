import settings

from train import Train_Runner


dataname = 'SYN-VAR'

settings = settings.get_settings(dataname)

runner = Train_Runner(settings)

runner.erun()