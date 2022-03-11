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

reviews = load_dataset("amazon_us_reviews", "Video_Games_v1_00", split="train[:8000]")
result = {}
result_2 = {}
for example in reviews:
    pid = example['product_id']
    body = example['review_body']
    body = body.replace('<br />', '')
    if pid in result:
        result_2[pid] += 1
        l = result[pid]
        result[pid] = l + [body]
    else:
        result[pid] = [body]
        result_2[pid] = 1
data = []

string_to_number = {}
for pid in result:
    r = result[pid]
    if len(r) > 1 and len(data) < 1000:
        reviews_concat = ' '.join(r)
        string_to_number[reviews_concat] = result_2[pid]
        data.append(reviews_concat)

    
#Find the average number of words in each review *CHANGE THIS MOFO*
total_words = 0
for review in string_to_number:
    summary_words = review.split(" ")
    summary_len = len(summary_words)
    words = int(summary_len/string_to_number[review])
    total_words += words

avg_num_words = int(total_words/len(string_to_number))

#Create pseudosummaries using summarizer
print("Creating pseudosummaries")
new_summaries = {}
count = 0
for doc in tqdm(string_to_number):
  count += 1
  if count == 10000:
    break
  summary_words = doc.split(" ")
  summary_len = len(summary_words)
  num_words = int(summary_len/string_to_number[doc])
  new_summary = summarizer.summarize(doc, words=num_words)
  new_summaries[doc] = new_summary


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
    #check this
    print(data[start + idx])
    new_sums.append(en)
    sum_docs.append(data[start + idx])