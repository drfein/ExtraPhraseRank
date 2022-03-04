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
total_words = 0
for sum in summaries:
    summary_words = sum.split()
    summary_len = len(sum)
    total_words += summary_len

avg_num_words = int(total_words/len(summaries))

#Create pseudosummaries using summarizer
print("Creating pseudosummaries")
new_summaries = []
count = 0
for doc in tqdm(documents):
  count += 1
  if count == 10000:
    break
  new_summary = summarizer.summarize(doc, words=avg_num_words)
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
        with open('json_data_' + str(count) + '.json', 'w') as outfile:
            json.dump(dictionary, outfile)
    
    de = str(en_to_de(sum)[0]['translation_text'])
    en = de_to_en(de)[0]['translation_text']
    print(en)
    print('$' * 30)
    print(documents[start + idx])
    new_sums.append(en)
    sum_docs.append(documents[start + idx])