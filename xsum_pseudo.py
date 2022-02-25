from ast import Assert
from sre_constants import ASSERT
from summa import summarizer
from BackTranslation import BackTranslation
from datasets import load_dataset 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
import time


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
trans = BackTranslation()
count = 400
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

  tries = 0
  while True:
    try:
      tries += 1
      if tries > 20:
        print("Failed to translate, waiting to restart")
        time.sleep(60)
        print("Restarting")
      paraphrased_sum = trans.translate(sum, src='en', tmp = 'es')
    except:
         continue
    else:
         break
  new_sums.append(str(paraphrased_sum))
  sum_docs.append(documents[start + idx])