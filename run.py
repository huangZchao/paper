import settings

from train import Train_Runner


dataname = 'ia-workplace-contacts'

settings = settings.get_settings(dataname, 6)  # todo seq_len

runner = Train_Runner(settings)

runner.erun()