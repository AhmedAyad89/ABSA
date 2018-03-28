import csv


def parse_hannah_legend(path):
	with open(path, 'r', encoding='mbcs') as fp:
		reader = csv.reader(fp, delimiter=',')
		listed = [x for x in reader]
	entity_dict = {}
	attr_dict = {}
	for i, line in enumerate(listed):
		if line[0] == 'entity':
			entity_dict[line[1]] = line[2]
		if line[0] == 'attribute':
			attr_dict[line[1]] = line[2]
	entity_dict['NA'] = 'not relevant'
	attr_dict['NA'] = 'not relevant'
	entity_dict['rel'] = 'relevant'
	attr_dict['rel'] = 'relevant'
	attr_dict[''] = 'general'  # added because sometimes no attr attached
	return entity_dict, attr_dict

def parse_hannah_csv(path):
	data = []
	attr = []
	entities = []
	num = []
	relevance = []
	sentiments = []
	with open(path,'r', encoding='mbcs') as fp:
		reader = csv.reader(fp, delimiter=',')

		listed = [x for x in reader ]

		counter=0
		for i,line in enumerate(listed):
			if line[0] == '***********':
				continue
			if not line[6]:
				continue
			num.append(line[0])
			data.append(line[6])
			attr.append([line[5]])
			entities.append([line[4]])
			sentiments.append([line[3]])
			relevance.append(line[2])
			j = i
			while (not listed[j + 1][0]):
				j += 1
				#print(j)
				entities[counter].append(listed[j][4])
				attr[counter].append(listed[j][5])
				sentiments[counter].append(listed[j][3])
			counter += 1

	print(len(data))
	labels = [None] * len(data)
	for i in range(len(data)):
		if relevance [i] == '9' or relevance [i] == 'g':
			if entities[i] == ['']:
				assert attr[i] == ['']
				entities[i] = ['rel']
				attr[i] = ['rel']
				sentiments[i] =['rel']
		else:
			entities[i] = ['NA']
			attr[i] = ['NA']
			sentiments[i] = ['NA']
		assert len(entities[i]) == len(attr[i])
		#print(data[i], relevance[i], entities[i], attr[i])
		labels[i] = list(zip(entities[i], attr[i], sentiments[i]))

	return data, labels

def hannah_merged_attr(attr_dict):
	attr_dict['ll'] = attr_dict['q']
	attr_dict['s'] = attr_dict['q']
	attr_dict['l'] = attr_dict['q']
	attr_dict['av'] = attr_dict['q']
	attr_dict['t'] = attr_dict['q']

	attr_dict['c'] = attr_dict['h']

	attr_dict['a'] = attr_dict['e']
	attr_dict['pp'] = attr_dict['e']
	return attr_dict

def hannah_merged_entity(entity_dict):
	entity_dict['p'] = entity_dict['g']
	entity_dict['f'] = entity_dict['g']
	entity_dict['c'] = entity_dict['g']

	entity_dict['cp'] = entity_dict['cg']
	entity_dict['cf'] = entity_dict['cg']
	entity_dict['cc'] = entity_dict['cg']
	return entity_dict

#maps from the shorthand labels to full labels + merged labels
def hannah_map_labels(labels, entity_dict, attr_dict, sent_dict):
	full_labels = []
	merged_labels = []
	for i, label in enumerate(labels):
		new_label = []
		for l in label:
			try:
				full = [entity_dict[l[0]], attr_dict[l[1]], sent_dict[l[2]] ]
			except Exception as e:
				print(e, ' : ', l)
				full = ['relevant', 'relevant', 'relevant']
			new_label.append(full)
		full_labels.append(new_label)

	attr_dict = hannah_merged_attr(attr_dict)
	entity_dict = hannah_merged_entity(entity_dict)

	for i, label in enumerate(labels):
		new_label = []
		for l in label:
			try:
				full = [entity_dict[l[0]], attr_dict[l[1]], sent_dict[l[2]]]
			except Exception as e:
				print(e, ' : ', l)
				full = ['relevant', 'relevant', 'relevant']
			new_label.append(full)
		merged_labels.append(new_label)

	return full_labels, merged_labels

if __name__ == '__main__':

	entity_dict, attr_dict = parse_hannah_legend('legend.csv')
	print(entity_dict)
	print(attr_dict)
	data, labels = parse_hannah_csv('19-03-05_comments_HD.csv')

	sent_dict = { 'p': 'Positive', 'n': 'Negative', '0':'Neutral',
								'':'not relevant', 'NA':'not relevant', 'rel': 'relevant' }

	full_labels, merged_labels = hannah_map_labels(labels, entity_dict, attr_dict, sent_dict)
	for i in range(6000):
		print(data[i])
		print(full_labels[i])
		print(merged_labels[i])
		print('\n-----------------\n')