from torch.utils.data import DataLoader
import math
import json
from datasets import load_dataset
from sklearn import preprocessing
import pandas as pd
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

sts_dataset_path = 'datasets/sickr-sts.tsv.gz'
dataset = load_dataset("mteb/sickr-sts")
numRow = dataset["test"].num_rows
print(numRow)
data = pd.DataFrame(dataset)

if not os.path.exists(sts_dataset_path):
    data.to_json('datasets/sickr-sts.tsv.gz')
model_name = 'bert-base-uncased'

# Read the dataset
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_sickr_sts_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read sickr-sts train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = json.load(fIn)  
    for i in range(numRow):  
        score = reader["test"][str(i)]["score"] / 5.0 
        inp_example = InputExample(texts=[reader["test"][str(i)]['sentence1'], reader["test"][str(i)]['sentence2']], label=score)
        if (i<numRow/5):
            test_samples.append(inp_example)
            dev_samples.append(inp_example)
        else:    
            train_samples.append(inp_example)
           
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read sickr-sts dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train datasets for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)

 