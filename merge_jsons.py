import json

datasets = []
counts = [1000, 1800]
for i in counts:
  with open('json_data_' + str(i) + '.json') as json_file:
    data = json.load(json_file)
    datasets.append(data)

datasetDict = {'document': [], 'summary': []}
for data in datasets:
    datasetDict['document'] += data['document']
    datasetDict['summary'] += data['summary']


# Save as json file
with open('merged_json_data.json', 'w') as outfile:
    json.dump(datasetDict, outfile)

