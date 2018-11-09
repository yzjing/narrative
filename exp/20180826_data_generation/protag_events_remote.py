from eventmakerTryServer_coref_remote import eventMaker
from collections import Counter, OrderedDict
import os

data_path = '../../data/raw/MovieSummaries/indiv_summaries'


files = [item for item in os.listdir(data_path) if '.txt' in item]
finished = [item for item in os.listdir('../../data/protag_events_gen_v_obj')]
finished = set([item.split('_')[0] for item in finished])

def rm_dups(elist):
	i = 0
	for j in range(1000):
		if elist[i] == elist[i+1]:
			del elist[i]
		i += 1
		if i == len(elist)-1:
			return elist

for name in files:
	if name.split('.')[0] not in finished:
		try:
			file_path = os.path.join(data_path, name)
			with open(file_path, 'r') as f:
		  		line = f.read()
			maker = eventMaker(line)
			maker.getEvent()
	#			for event in maker.events:
	#				print(event)
	#			print()
			events = maker.events
			#Remove duplicates
			events = rm_dups(events)
			subs = [event[0] for event in events]
			c = Counter(subs)
			# Use the character that appears most as the protagonist
			protag = c.most_common(1)[0][0]
			fil_events = [event for event in maker.events if event[0] == protag or event[2] == protag]
			with open('../../data/protag_events_nogen/' + name.split('.')[0] + '_protag_events.txt', 'w') as f:
				for event in fil_events:
					sentence = " ".join(event)
					f.write(sentence)
					f.write('\n')
		except:
			pass
