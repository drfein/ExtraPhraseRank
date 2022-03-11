from ast import Assert
from sre_constants import ASSERT
from datasets import load_dataset 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from tqdm.auto import tqdm
import time
from transformers import pipeline
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')


print("Getting dataset")
xsum_data = load_dataset("xsum", split="train[:2]")

loader = DataLoader(xsum_data, batch_size=len(xsum_data))
x_sum = next(iter(loader))
documents = x_sum['document']
summaries = x_sum['summary']

#Find the average number of words in dataset
sum_lengths = []
for sum in summaries:
    print("############################################################################")
    print("Summary: " + sum)
    print("############################################################################")
    summary_words = sum.split(" ")
    summary_len = len(summary_words)
    sum_lengths.append(summary_len)

#Create pseudosummaries using heuristic
print("Creating pseudosummaries")

def grab_first(doc, words):
    words_left = words
    # Break document into sentences
    sentences = nltk.tokenize.sent_tokenize(doc)

    # Grab the number of sentences that are less than the number of words
    result_sentences = []
    for sentence in sentences:
        sent_len = len(sentence.split())
        if sent_len <= words_left:
            result_sentences.append(sentence)
            words_left -= sent_len
        else:
            if sent_len < 2 * words_left:
                result_sentences.append(sentence)
                break
    # Join the sentences together
    summary = ' '.join(result_sentences)
    return summary

new_summaries = []
count = 0
for idx, doc in tqdm(enumerate(documents)):
  count += 1
  if count == 10000:
    break
  new_summary = grab_first(doc, sum_lengths[idx])
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
    print("Pseudo-Summary: " + en)
    print("############################################################################")
    print("Document: " + documents[start + idx])
    print("############################################################################")
    new_sums.append(en)
    sum_docs.append(documents[start + idx])