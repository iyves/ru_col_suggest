
from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils

class SentencesLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        for line in open(self.data_path, "r", encoding="utf-8"):
            yield utils.simple_preprocess(line)

class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t', flush=True)

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            print(f'  Loss after epoch {self.epoch}: {loss}', flush=True)
        else:
            print(f'  Loss after epoch {self.epoch}: {loss - self.loss_previous_step}', flush=True)
        self.losses.append(loss)
        self.epoch += 1
        self.loss_previous_step = loss