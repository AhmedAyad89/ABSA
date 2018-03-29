import json
from Features.Openai_transform import *


with open('Dataset/Semeval_laptop_train.json', 'r') as infile:
	json_dict = json.load(infile)

openai_transform(json_dict, 'sentence')

