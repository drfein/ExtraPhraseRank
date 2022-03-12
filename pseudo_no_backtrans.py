from ast import Assert
from sre_constants import ASSERT
from summa import summarizer
from datasets import load_dataset 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
import time
from transformers import pipeline


print("Getting dataset")
xsum_data = load_dataset("xsum", split="train[:1000]")

loader = DataLoader(xsum_data, batch_size=len(xsum_data))
x_sum = next(iter(loader))
documents = x_sum['document']
summaries = x_sum['summary']

#Find the average number of words in dataset
sum_lengths = []
for sum in summaries:
    # print("############################################################################")
    # print("Summary: " + sum)
    # print("############################################################################")
    summary_words = sum.split(" ")
    summary_len = len(summary_words)
    sum_lengths.append(summary_len)

#Create pseudosummaries using summarizer
print("Creating pseudosummaries")
new_summaries = []
count = 0
for idx, doc in tqdm(enumerate(documents)):
  count += 1
  if count == 10000:
    break
  new_summary = summarizer.summarize(doc, words=sum_lengths[idx])
  new_summaries.append(new_summary)

# Backtranslate the sentences
print("Backtranslating")
en_to_de = pipeline('translation', model='Helsinki-NLP/opus-mt-en-de')
de_to_en = pipeline('translation', model='Helsinki-NLP/opus-mt-de-en')
count = 0
start = count
sum_docs = []
new_sums = []
for idx, sum in tqdm(enumerate(new_summaries[start:])):
    count += 1

    # set to the number of examples
    if count % 200 == 0:
        dictionary = {'document': sum_docs, 'summary': new_sums}
        with open('json_extrarank_' + str(count) + '.json', 'w') as outfile:
            json.dump(dictionary, outfile)
    new_sums.append(sum)
    sum_docs.append(documents[start + idx])