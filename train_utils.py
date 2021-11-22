
from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils

from transformers import LineByLineTextDataset # for loading in dataset
from transformers import DataCollatorForLanguageModeling # for batching
from transformers import Trainer, TrainingArguments

class SentencesLoader:
    def __init__(self, data_path, preprocess=True):
        self.data_path = data_path
        self.preprocess = preprocess

    def __iter__(self):
        for line in open(self.data_path, "r", encoding="utf-8"):
            if self.preprocess:
                yield utils.simple_preprocess(line)
            else:
                yield line

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


def train_bert(path, tokenizer, model_save_dir, model, epochs=5):
    '''trains a bert model'''
    # At this moment, this class does not allow for loading multiple files at one :/
    print(f"Loading {path}", flush=True)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=path,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print(f"Finished loading {path}", flush=True)

    training_args = TrainingArguments(
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=32, # Restart runtime & modify this if GPU crashes from low memory: 32,16,8,4,1
        save_steps=10_000,
        save_total_limit=2,
        dataloader_num_workers=40
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    print(f"Saving model at `{model_save_dir}`")
    trainer.save_model(model_save_dir)

    print(f"Finished training for `{model_save_dir}`!")