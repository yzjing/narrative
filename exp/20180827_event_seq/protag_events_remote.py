from eventmakerTryServer_coref_remote import eventMaker
from collections import Counter
import os

data_path = '../../data/MovieSummaries/indiv_summaries'


files = [item for item in os.listdir(data_path) if '.txt' in item]
finished = [item for item in os.listdir('protag_events/')]
finished = [item.split('_')[0] for item in finished]


for name in files:
	if name.split('.')[0] not in finished:
		try:
			file_path = os.path.join(data_path, name)
			with open(file_path, 'r') as f:
		  		line = f.read()
			maker = eventMaker(line)
			maker.getEvent()
			subs = [event[0] for event in maker.events]
			c = Counter(subs)
			# Use the character that appears most as the protagonist
			protag = c.most_common(1)[0][0]
			fil_events = [event for event in maker.events if event[0] == protag]
			with open('protag_events/' + name.split('.')[0] + '_protag_events.txt', 'w') as f:
				for event in fil_events:
					sentence = " ".join(event)
					f.write(sentence)
					f.write('\n')
		except:
			pass
