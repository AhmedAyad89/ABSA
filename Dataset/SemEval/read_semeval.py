import pickle
import spacy
import xml.etree.ElementTree as ET
import operator



#data is a list(reviews) of lists(sentences)
#labels is a list(reviews) of lists(sentences) of lists(opinions)
#data is pickled as text and spacy tokens
def pickle_dataset(name, filename):
	tree = ET.parse(filename)
	root = tree.getroot()
	#nlp = spacy.load('en_core_web_lg')
	data = []
	labels = []
	label_set={}
	entities =[]
	attributes =[]
	for r in root.findall('Review/sentences'):
		review = []
		for sentence in r:
			for text in sentence.findall('text'):
				review.append(text.text.strip())
		data.append(review)


	for r in root.findall('Review/sentences'):
		review = []
		for sentence in r:
			s = []
			for cat in sentence.findall('Opinions/Opinion'):
				#print(cat.attrib)
				category = cat.attrib['category'].split('#')
				s.append([category[0], category[1], cat.attrib['polarity'] ])
				print(s[-1])
				if(category[0] not in label_set):
					label_set[category[0]] ={}
					entities.append(category[0])
				else:
					if(category[1] not in label_set[category[0]]):
						label_set[category[0]][category[1]]=1
						if(category[1] not in attributes):
							attributes.append(category[1])
					else:
						label_set[category[0]][category[1]] += 1
			if(s == []):
				s = [['NA', 'NA', 'NA']]
			review.append(s)
		labels.append(review)

	attributes.append('NA')
	entities.append('NA')

	for key in label_set.keys():
		print(key)
		print(label_set[key])
		print('\n----------\n')
	with open(name+'-rawTextData', 'wb') as fp:
		pickle.dump(data, fp)
	with open(name+'-rawTextLabels', 'wb') as fp:
		pickle.dump(labels, fp)

	with open(name+'-LabelCounts', 'wb') as fp:
		pickle.dump(label_set, fp)
	with open(name+'-Entities', 'wb') as fp:
		pickle.dump(entities, fp)
	with open(name+'-Attributes', 'wb') as fp:
		pickle.dump(attributes, fp)

	# print(labels)
	# for review in data:
	# 	for sentence in review:
	# 		sentence = nlp(sentence)
	# print(data)
	# with open('semevalLaptop-spacyTextData', 'wb') as fp:
	# 	pickle.dump(data, fp)




if __name__ == '__main__':
	print('x')
	pickle_dataset('semevalLaptop_test', 'EN_LAPT_SB1_TEST_.xml.gold')
	pickle_dataset('semevalRestaurant_test', 'EN_REST_SB1_TEST.xml.gold')


