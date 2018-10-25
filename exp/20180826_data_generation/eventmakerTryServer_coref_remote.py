
# This is a class to extract events from sentences, store words in a working memory and ?translate back to human readable 
# sentences from partially generalized sentences.
from nltk.tag.stanford import StanfordNERTagger
import nltk.tag.stanford as stanford
from nltk.internals import find_jars_within_path
from nltk import word_tokenize, pos_tag, DependencyGraph, FreqDist
from nltk.tree import Tree
from nltk.internals import find_jar, find_jar_iter, config_java, java, _java_options, find_jars_within_path
#from en import verb
import os, copy
from collections import defaultdict
import nltk.corpus
from nltk.corpus import wordnet as wn
import re
from pprint import pprint
import tempfile
from nltk import compat
from subprocess import PIPE
import subprocess
import json
from numpy.random import choice
from nltk.wsd import lesk

# necessary parameters
stanford_dir = "../../pipeline/stanford-corenlp-full-2016-10-31"
models = "../../pipeline/stanford-english-corenlp-2016-10-31-models"
ner_loc = "../../pipeline/stanford-ner-2016-10-31"
verbnet = nltk.corpus.VerbnetCorpusReader('../../pipeline/verbnet', ['absorb-39.8.xml','continue-55.3.xml','hurt-40.8.3.xml','remove-10.1.xml','accept-77.xml','contribute-13.2.xml','illustrate-25.3.xml','render-29.90.xml','accompany-51.7.xml','convert-26.6.2.xml','image_impression-25.1.xml','require-103.xml','acquiesce-95.xml','cooking-45.3.xml','indicate-78.xml','resign-10.11.xml','addict-96.xml','cooperate-73.xml','inquire-37.1.2.xml','risk-94.xml','adjust-26.9.xml','cope-83.xml','instr_communication-37.4.xml','roll-51.3.1.xml','admire-31.2.xml','correlate-86.1.xml','interrogate-37.1.3.xml','rummage-35.5.xml','admit-65.xml','correspond-36.1.xml','investigate-35.4.xml','run-51.3.2.xml','adopt-93.xml','cost-54.2.xml','involve-107.xml','rush-53.2.xml','advise-37.9.xml','crane-40.3.2.xml','judgment-33.xml','say-37.7.xml','allow-64.xml','create-26.4.xml','keep-15.2.xml','scribble-25.2.xml','amalgamate-22.2.xml','curtsey-40.3.3.xml','knead-26.5.xml','search-35.2.xml','amuse-31.1.xml','cut-21.1.xml','learn-14.xml','see-30.1.xml','animal_sounds-38.xml','debone-10.8.xml','leave-51.2.xml','seem-109.xml','appeal-31.4.xml','declare-29.4.xml','lecture-37.11.xml','send-11.1.xml','appear-48.1.1.xml','dedicate-79.xml','light_emission-43.1.xml','separate-23.1.xml','appoint-29.1.xml','deduce-97.2.xml','limit-76.xml','settle-89.xml','assessment-34.1.xml','defend-72.2.xml','linger-53.1.xml','shake-22.3.xml','assuming_position-50.xml','destroy-44.xml','sight-30.2.xml','avoid-52.xml','devour-39.4.xml','lodge-46.xml','simple_dressing-41.3.1.xml','banish-10.2.xml','differ-23.4.xml','long-32.2.xml','slide-11.2.xml','base-97.1.xml','dine-39.5.xml','manner_speaking-37.3.xml','smell_emission-43.3.xml','battle-36.4.xml','disappearance-48.2.xml','marry-36.2.xml','snooze-40.4.xml','become-109.1.xml','disassemble-23.3.xml','marvel-31.3.xml','beg-58.2.xml','discover-84.xml','masquerade-29.6.xml','sound_emission-43.2.xml','begin-55.1.xml','dress-41.1.1.xml','matter-91.xml','sound_existence-47.4.xml','being_dressed-41.3.3.xml','dressing_well-41.3.2.xml','meander-47.7.xml','spank-18.3.xml','bend-45.2.xml','drive-11.5.xml','meet-36.3.xml','spatial_configuration-47.6.xml','benefit-72.1.xml','dub-29.3.xml','mine-10.9.xml','spend_time-104.xml','berry-13.7.xml','eat-39.1.xml','mix-22.1.xml','split-23.2.xml','bill-54.5.xml','empathize-88.2.xml','modes_of_being_with_motion-47.3.xml','spray-9.7.xml','body_internal_motion-49.xml','enforce-63.xml','multiply-108.xml','stalk-35.3.xml','body_internal_states-40.6.xml','engender-27.xml','murder-42.1.xml','steal-10.5.xml','braid-41.2.2.xml','ensure-99.xml','neglect-75.xml','stimulus_subject-30.4.xml','break-45.1.xml','entity_specific_cos-45.5.xml','nonvehicle-51.4.2.xml','stop-55.4.xml','breathe-40.1.2.xml','entity_specific_modes_being-47.2.xml','nonverbal_expression-40.2.xml','subjugate-42.3.xml','bring-11.3.xml','equip-13.4.2.xml','obtain-13.5.2.xml','substance_emission-43.4.xml','build-26.1.xml','escape-51.1.xml','occurrence-48.3.xml','succeed-74.xml','bulge-47.5.3.xml','establish-55.5.xml','order-60.xml','suffocate-40.7.xml','bump-18.4.xml','estimate-34.2.xml','orphan-29.7.xml','suspect-81.xml','butter-9.9.xml','exceed-90.xml','other_cos-45.4.xml','sustain-55.6.xml','calibratable_cos-45.6.xml','exchange-13.6.xml','overstate-37.12.xml','swarm-47.5.1.xml','calve-28.xml','exhale-40.1.3.xml','own-100.xml','swat-18.2.xml','captain-29.8.xml','exist-47.1.xml','pain-40.8.1.xml','talk-37.5.xml','care-88.1.xml','feeding-39.7.xml','patent-101.xml','tape-22.4.xml','carry-11.4.xml','ferret-35.6.xml','pay-68.xml','tell-37.2.xml','carve-21.2.xml','fill-9.8.xml','peer-30.3.xml','throw-17.1.xml','change_bodily_state-40.8.4.xml','fire-10.10.xml','pelt-17.2.xml','tingle-40.8.2.xml','characterize-29.2.xml','fit-54.3.xml','performance-26.7.xml','touch-20.xml','chase-51.6.xml','flinch-40.5.xml','pit-10.7.xml','transcribe-25.4.xml','cheat-10.6.xml','floss-41.2.1.xml','pocket-9.10.xml','transfer_mesg-37.1.1.xml','chew-39.2.xml','focus-87.1.xml','poison-42.2.xml','try-61.xml','chit_chat-37.6.xml','forbid-67.xml','poke-19.xml','turn-26.6.1.xml','classify-29.10.xml','force-59.xml','pour-9.5.xml','urge-58.1.xml','clear-10.3.xml','free-80.xml','preparing-26.3.xml','use-105.xml','cling-22.5.xml','fulfilling-13.4.1.xml','price-54.4.xml','vehicle-51.4.1.xml','coil-9.6.xml','funnel-9.3.xml','promise-37.13.xml','vehicle_path-51.4.3.xml','coloring-24.xml','future_having-13.3.xml','promote-102.xml','complain-37.8.xml','get-13.5.1.xml','pronounce-29.3.1.xml','complete-55.2.xml','give-13.1.xml','push-12.xml','void-106.xml','comprehend-87.2.xml','gobble-39.3.xml','put-9.1.xml','waltz-51.5.xml','comprise-107.1.xml','gorge-39.6.xml','put_direction-9.4.xml','want-32.1.xml','concealment-16.xml','groom-41.1.2.xml','put_spatial-9.2.xml','weather-57.xml','confess-37.10.xml','grow-26.2.xml','reach-51.8.xml','weekend-56.xml','confine-92.xml','help-72.xml','reflexive_appearance-48.1.2.xml','wink-40.3.1.xml','confront-98.xml','herd-47.5.2.xml','refrain-69.xml','wipe_instr-10.4.2.xml','conjecture-29.5.xml','hiccup-40.1.1.xml','register-54.1.xml','wipe_manner-10.4.1.xml','consider-29.9.xml','hire-13.5.3.xml','rehearse-26.8.xml','wish-62.xml','conspire-71.xml','hit-18.1.xml','relate-86.2.xml','withdraw-82.xml','consume-66.xml','hold-15.1.xml','rely-70.xml','contiguous_location-47.8.xml','hunt-35.1.xml','remedy-45.7.xml'])


class eventMaker: 
	def __init__(self,sentence):
		self.sentence = sentence
		self.nouns = defaultdict(list)
		self.verbs = defaultdict(list)
		self.events = []


	def lookupNoun(self,word, pos, original_sent):
		#print(word, pos)
		# This is a function that supports generalize_noun function 
		if len(wn.synsets(word)) > 0:
			#word1 = lesk(original_sent.split(), word, pos='n')  #word1 is the first synonym of word
			#print(word1)
			return str(lesk(original_sent.split(), word, pos='n'))
		else:
			return word.lower()

	def lookupAdj(self,word, pos, original_sent):
		#print(word, pos)
		# This is a function that supports generalize_noun function 
		if len(wn.synsets(word)) > 0:
			#word1 = lesk(original_sent.split(), word, pos='n')  #word1 is the first synonym of word
			#print(word1)
			return str(lesk(original_sent.split(), word, pos='a'))
		else:
			return word.lower()	


	def generalize_noun(self, word, tokens, named_entities, original_sent):
		# This function is to support getEvent functions. Tokens have specific format(lemma, pos, ner)
		lemma = tokens[word][0]
		pos = tokens[word][1]
		ner = tokens[word][2]
		resultString = ""

		if ner != "O": # output of Stanford NER: default values is "O"
			if ner == "PERSON":
				if word not in named_entities: # named_entities is a list to store the names of people
					named_entities.append(word)
				resultString = "<NE>"+str(named_entities.index(word))
			else:
				resultString = ner 
		else:
			word = lemma
			if "NN" in pos: # and adjective? #changed from only "NN"
				resultString = self.lookupNoun(word, pos, original_sent) # get the word's ancestor
			elif "JJ" in pos:
				resultString = self.lookupAdj(word, pos, original_sent)
			elif "PRP" in pos:
				if word == "he" or word == "him":
					resultString = "Synset('male.n.02')"
				elif word == "she" or word == "her":
					resultString = "Synset('female.n.02')"
				elif word == "I" or word == "me" or word == "we" or word == "us":
					resultString = "Synset('person.n.01')"
				elif word == "they" or word == "them":
					resultString = "Synset('physical_entity.n.01')"
				else:
					resultString = "Synset('entity.n.01')" 
			else:
				resultString = word
		return resultString, named_entities


	def generalize_verb(self,word,tokens):
		# This function is to support getEvent functions. Tokens have specific format:tokens[word] = [lemma, POS, NER]
		word = tokens[word][0]
		if word == "have": return "own-100" 

		classids = verbnet.classids(word)
		if len(classids) > 0:
			#return choice based on weight of number of members
			mems = []
			for classid in classids:
				vnclass = verbnet.vnclass(classid)
				num = len(list(vnclass.findall('MEMBERS/MEMBER')))
				mems.append(num)
			mem_count = mems
			mems = [x/float(sum(mem_count)) for x in mems]
			return str(choice(classids, 1, p=mems)[0])
		else:
			return word

	def build_coref_dic(self, dependencies):
		corefs = dependencies["corefs"]
		coref_dict = {}
		for i in corefs:
			# print(i)
			# print(corefs[i])
			for item in corefs[i]:
				# print(item)
				cd = dict(item)
				#We want the representative one to be the name, not pronoun.
				if cd["isRepresentativeMention"] == False and cd['type'] == 'PROPER':
					cd["isRepresentativeMention"] == True

				if cd["isRepresentativeMention"] == True and cd['type'] != 'PROPER':
					cd["isRepresentativeMention"] == False

				if cd["isRepresentativeMention"] == True:
					v = (cd['sentNum'], cd['headIndex'])

			for item in corefs[i]:
				cd = dict(item)
				if cd['gender'] == 'MALE' or cd['gender'] == 'FEMALE':
					if cd["isRepresentativeMention"] == False:
						coref_dict[(cd['sentNum'], cd['headIndex'])] = v
		return coref_dict

	def check_coref(self, coref_dict, sentid, tokenid, dependencies):
		if (sentid, tokenid) in coref_dict.keys():
			coref_sent_idx, coref_token_idx = coref_dict[(sentid, tokenid)]
			# print(coref_sent_idx, coref_token_idx)
			coref_word = dependencies[coref_sent_idx-1]['tokens'][coref_token_idx-1]['word']
			return coref_word
		else:
			return False

	def callStanford(self,sentence):
		# This function can call Stanford CoreNLP tool and support getEvent function.
		encoding = "utf8"
		cmd = ["java", "-cp", stanford_dir+"/*","-Xmx20g", "edu.stanford.nlp.pipeline.StanfordCoreNLPClient",
			"-annotators", "tokenize,ssplit,parse,ner,pos,lemma,depparse,coref", #ner,tokenize,ssplit,pos,lemma,parse,depparse #tokenize,ssplit,pos,lemma,depparse,ner
			'-outputFormat','json',
			"-parse.flags", "",
			'-encoding', encoding,
			'-model', models+'/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz',"-backends","localhost:9000,12"]

		input_ = ""
		default_options = ' '.join(_java_options)
		with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
			# Write the actual sentences to the temporary input file
			if isinstance(sentence, compat.text_type) and encoding:
				input_ = sentence.encode(encoding)
			input_file.write(input_)
			input_file.flush()
			input_file.seek(0)
			#print(input_file)
			devnull = open(os.devnull, 'w')
			out = subprocess.check_output(cmd, stdin=input_file)
			out = out.replace(b'\xc2\xa0',b' ')
			out = out.replace(b'\xa0',b' ')
			out = out.replace(b'NLP>',b'')
			#print(out)
			out = out.decode(encoding)
			#print(out)
		os.unlink(input_file.name)
		# Return java configurations to their default values.
		config_java(options=default_options, verbose=False)
		return out


	def getEvent(self):
	# This is a function that can extract the event format from the sentence given.
		words = self.sentence.split()
		original_sent = self.sentence.strip()

		json_data = self.callStanford(self.sentence)
		# print(json_data)
		d = json.loads(json_data)
		all_json = d["sentences"]
		ner_dict2 = {}
		for i, sentence in enumerate(all_json):
			for token in sentence["tokens"]:
				ner_dict2[token["word"]] = token["ner"]

		coref_dict = self.build_coref_dic(d)
	
		for sent_num, sentence in enumerate(all_json): # for each sentence in the entire input
			tokens = defaultdict(list) 

			for token in sentence["tokens"]: 
				tokens[token["word"]] = [token["lemma"], token["pos"], ner_dict2[token["word"]],token["index"]] # each word in the dictionary has a list of [lemma, POS, NER]

			deps = sentence["enhancedPlusPlusDependencies"]	 # retrieve the dependencies
			named_entities = []
			verbs = []
			subjects = []
			modifiers = []
			objects = []
			pos = {}
			pos["EmptyParameter"] = "None"
			chainMods = {} # chaining of mods
			index = defaultdict(list)  #for identifying part-of-speech
			index["EmptyParameter"] = -1

			# create events
			for token_id, d in enumerate(deps):
				if 'nsubj' in d["dep"] and "RB" not in tokens[d["dependentGloss"]][1]: #adjective? #"csubj" identifies a lot of things wrong 
					#print(tokens[d["dependentGloss"]][1])
					if d["governorGloss"] not in verbs:
						#create new event
						if not "VB" in tokens[d["governorGloss"]][1]: continue
						verbs.append(d["governorGloss"])
						index[d["governorGloss"]] = d["governor"] #adding index
						# Here sent_num starts from 0, but in the coref data starts from 1.
						# token_id starts from 0, but in the token data starts from 1?
						coref = self.check_coref(coref_dict, sent_num+1, token_id, all_json)
						if coref:
							subjects.append(coref)
						else:
							subjects.append(d["dependentGloss"])
						index[d["dependentGloss"]] = d["dependent"] #adding index to subject
						pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						modifiers.append('EmptyParameter')
						objects.append('EmptyParameter')
					elif d["governorGloss"] in verbs:
						if subjects[verbs.index(d["governorGloss"])] == "EmptyParameter": # if verb alrady exist 
							# d["govenor"] is the token id of the governor
							coref = self.check_coref(coref_dict, sent_num+1, d["governor"], all_json)
							if coref:
								ubjects[verbs.index(d["governorGloss"])] = coref
							else:
								subjects[verbs.index(d["governorGloss"])] = d["dependentGloss"]
							index[d["dependentGloss"]] = d["dependent"]
						else:
							coref = self.check_coref(coref_dict, sent_num+1, d["dependent"], all_json)
							if coref:
								subjects.append(coref)
							else:
								subjects.append(d["dependentGloss"])
							index[d["dependentGloss"]] = d["dependent"]
							verbs.append(d["governorGloss"])
							index[d["governorGloss"]] = d["governor"]
							modifiers.append('EmptyParameter')
							objects.append('EmptyParameter')
						pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					elif d["dependentGloss"] in subjects: # one subject multiple verbs
						verbs[subjects.index(d["dependentGloss"])] = d["governorGloss"]
						index[d["governorGloss"]] = d["governor"]
						pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
				else: #check to see if we have a subject filled ??
					if len(subjects) >1:
						if subjects[-1] == "EmptyParameter":
							subjects[-1] = subjects[-2]
					#conjunction of verbs
					if 'conj' in d["dep"] and 'VB' in tokens[d["dependentGloss"]][1]:
						if d["dependentGloss"] not in verbs:
							verbs.append(d["dependentGloss"])
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
							index[d["dependentGloss"]] = d["dependent"]
							subjects.append('EmptyParameter')
							modifiers.append('EmptyParameter')
							objects.append('EmptyParameter')
					#conjunction of subjects
					elif 'conj' in d["dep"] and d["governorGloss"] in subjects: # governor and dependent are both subj. e.g. Amy and Sheldon
						loc = subjects.index(d["governorGloss"])
						verb = verbs[loc] #verb already exist. question: should the verb have the same Part-of-Speech tag?
						coref = self.check_coref(coref_dict, sent_num+1, d["dependent"], all_json)
						if coref:
							subjects.append(coref)
						else:
							subjects.append(d["dependentGloss"])
						index[d["dependentGloss"]] = d["dependent"]
						verbs.append(verb)
						modifiers.append('EmptyParameter')
						objects.append('EmptyParameter')
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					elif 'conj' in d["dep"] and d["governorGloss"] in objects:
						loc = objects.index(d["governorGloss"])
						match_verb = verbs[loc] #??? is it a correct way to retrieve the verb?
						#print(match_verb)
						temp_verbs = copy.deepcopy(verbs)
						for i, verb in enumerate(temp_verbs):
							if match_verb == verb: # what if the verb appears more than one times?
								subjects.append(subjects[i]) 
								verbs.append(verb)
								modifiers.append('EmptyParameter')
								coref = self.check_coref(coref_dict, sent_num+1, d["dependent"], all_json)
								if coref:
									objects.append(coref)
								else:
									objects.append(d["dependentGloss"])
								index[d["dependentGloss"]] = d["dependent"]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]			
					# case 1: obj
					elif 'dobj' in d["dep"] or 'xcomp' == d["dep"]:  #?? 'xcomp' is a little bit tricky
						if d["governorGloss"] in verbs:
							#modify that object
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
							for i, verb in reversed(list(enumerate(verbs))):
								if verb == d["governorGloss"] and objects[i] == "EmptyParameter":
									coref = self.check_coref(coref_dict, sent_num+1, d["dependent"], all_json)
									if coref:
										objects.append(coref)
									else:
										objects[i] = d["dependentGloss"]
									index[d["dependentGloss"]] = d["dependent"]
					# case 2: nmod
					elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'NN' in tokens[d["dependentGloss"]][1]:
						if d["governorGloss"] in verbs: # how about PRP?
							#modify that modifier
							for i, verb in reversed(list(enumerate(verbs))):
								if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
									modifiers[i] = d["dependentGloss"]
									index[d["dependentGloss"]] = d["dependent"]
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						elif d["governorGloss"] in chainMods: # is not used actually
							v = chainMods[d["governorGloss"]]
							if v in verbs:
								modifiers[verbs.index(v)] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
								pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					# PRP			
					elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'PRP' in tokens[d["dependentGloss"]][1]:
						if d["governorGloss"] in verbs: # how about PRP?
							#modify that modifier
							for i, verb in reversed(list(enumerate(verbs))):
								if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
									modifiers[i] = d["dependentGloss"]
									index[d["dependentGloss"]] = d["dependent"]
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						elif d["governorGloss"] in chainMods: # is not used actually
							v = chainMods[d["governorGloss"]]
							if v in verbs:
								modifiers[verbs.index(v)] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
								pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]


				
			# generalize the words and store them in instance variables
			for (a,b,c,d) in zip(subjects, verbs, objects, modifiers):
				# pos1 = "None"
				# pos2 = "None"
				# pos3 = "None"
				# pos4 = "None"
				# poslabel = "None"
				# #print((a,b,c,d))
				# num = 0
				# if a != 'EmptyParameter':
				# 	if index[a] == tokens[a][-1]: #adding part-of-speech
				# 		pos1 = tokens[a][1]
				# 	a1, named_entities = self.generalize_noun(a, tokens, named_entities, original_sent)
				# 	if "<NE>" in a1:
				# 		self.nouns["<NE>"].append(tokens[a][0])
				# 	elif a1 == a:
				# 		a1 = self.generalize_verb(a, tokens) #changed line
				# 		self.verbs[a1].append(tokens[a][0])
				# 	else:
				# 		self.nouns[a1].append(tokens[a][0])
				# else:
				# 	a1 = a
				if b != 'EmptyParameter':
					if index[b] == tokens[b][-1]: #may have issue in looping
						pos2 = tokens[b][1]
					b1 = self.generalize_verb(b, tokens) #changed line
					self.verbs[b1].append(tokens[b][0])
				else:
					b1 = b
				#if c != 'EmptyParameter':
#					if index[c] == tokens[c][-1]:
#						pos3 = tokens[c][1]
#					c1, named_entities = self.generalize_noun(c, tokens, named_entities, original_sent)
#					if "<NE>" in c1:
#						self.nouns["<NE>"].append(tokens[c][0])
#					elif c1 == c:
#						c1 = self.generalize_verb(c, tokens) #changed line
#						self.verbs[c1].append(tokens[c][0])
#					else:
#						self.nouns[c1].append(tokens[c][0])
#				else:
#					c1 = c
#				if d == 'EmptyParameter': #adding new lines start. 
#					label = 'EmptyParameter'
#				else:
#					label = 'None' #change from Exist
#					for dep in deps:
#						if b == dep["governorGloss"] and d == dep["dependentGloss"] and "nmod" in dep["dep"]:
#							if ":" in dep["dep"]:
#								label = dep["dep"].split(":")[1]  # shoud this line be added??
#							num = dep['dependent']
#							#print(dep)
#							#print("number:")
#							#print(num)
#							#print(type(num))
#					for dep in deps: # how about obl dependency?
#						if b == dep["governorGloss"] and d == dep["dependentGloss"] and "obl" in dep["dep"]:
#							if ":" in dep["dep"]:
#								label = dep["dep"].split(":")[1]
#							num = dep['dependent']
#							#print(dep)
#								
#					for dep in deps:
#						if "case" in dep["dep"] and d == dep["governorGloss"] and num == dep['governor']: # #what if modifier is related to multiple labels?
#							label = dep["dependentGloss"]	#adding new lines end
#							index[label] = dep["dependent"]
#							#print("label:")
#							#print(num)
#							#print(dep['governor'])
#							#print(type(dep['governor']))
#				if d != 'EmptyParameter':
#					if index[d] == tokens[d][-1]:
#						pos4 = tokens[d][1]
#					d1, named_entities = self.generalize_noun(d, tokens, named_entities, original_sent)
#					if "<NE>" in d1:
#						self.nouns["<NE>"].append(tokens[d][0])
#					else:
#						self.nouns[d1].append(tokens[d][0])
#				else:
#					d1 = d	
#				if label != "EmtpyParameter" and label != "None":
#					if len(tokens[label]) == 4:
#						#print(tokens[label])
#						#print(index[label])
#						if tokens[label][-1] == index[label]:
#							poslabel = tokens[label][1]

				self.events.append([a,b1,c,d])

# line = '''The nation of Panem consists of a wealthy Capitol and twelve poorer districts. As punishment for a past rebellion, each district must provide a boy and girl  between the ages of 12 and 18 selected by lottery  for the annual Hunger Games. The tributes must fight to the death in an arena; the sole survivor is rewarded with fame and wealth. In her first Reaping, 12-year-old Primrose Everdeen is chosen from District 12. Her older sister Katniss volunteers to take her place. Peeta Mellark, a baker's son who once gave Katniss bread when she was starving, is the other District 12 tribute. Katniss and Peeta are taken to the Capitol, accompanied by their frequently drunk mentor, past victor Haymitch Abernathy. He warns them about the "Career" tributes who train intensively at special academies and almost always win. During a TV interview with Caesar Flickerman, Peeta unexpectedly reveals his love for Katniss. She is outraged, believing it to be a ploy to gain audience support, as "sponsors" may provide in-Games gifts of food, medicine, and tools. However, she discovers Peeta meant what he said. The televised Games begin with half of the tributes killed in the first few minutes; Katniss barely survives ignoring Haymitch's advice to run away from the melee over the tempting supplies and weapons strewn in front of a structure called the Cornucopia. Peeta forms an uneasy alliance with the four Careers. They later find Katniss and corner her up a tree. Rue, hiding in a nearby tree, draws her attention to a poisonous tracker jacker nest hanging from a branch. Katniss drops it on her sleeping besiegers. They all scatter, except for Glimmer, who is killed by the insects. Hallucinating due to tracker jacker venom, Katniss is warned to run away by Peeta. Rue cares for Katniss for a couple of days until she recovers. Meanwhile, the alliance has gathered all the supplies into a pile. Katniss has Rue draw them off, then destroys the stockpile by setting off the mines planted around it. Furious, Cato kills the boy assigned to guard it. As Katniss runs from the scene, she hears Rue calling her name. She finds Rue trapped and releases her. Marvel, a tribute from District 1, throws a spear at Katniss, but she dodges the spear, causing it to stab Rue in the stomach instead. Katniss shoots him dead with an arrow. She then comforts the dying Rue with a song. Afterward, she gathers and arranges flowers around Rue's body. When this is televised, it sparks a riot in Rue's District 11. President Snow summons Seneca Crane, the Gamemaker, to express his displeasure at the way the Games are turning out. Since Katniss and Peeta have been presented to the public as "star-crossed lovers", Haymitch is able to convince Crane to make a rule change to avoid inciting further riots. It is announced that tributes from the same district can win as a pair. Upon hearing this, Katniss searches for Peeta and finds him with an infected sword wound in the leg. She portrays herself as deeply in love with him and gains a sponsor's gift of soup. An announcer proclaims a feast, where the thing each survivor needs most will be provided. Peeta begs her not to risk getting him medicine. Katniss promises not to go, but after he falls asleep, she heads to the feast. Clove ambushes her and pins her down. As Clove gloats, Thresh, the other District 11 tribute, kills Clove after overhearing her tormenting Katniss about killing Rue. He spares Katniss "just this time...for Rue". The medicine works, keeping Peeta mobile. Foxface, the girl from District 5, dies from eating nightlock berries she stole from Peeta; neither knew they are highly poisonous. Crane changes the time of day in the arena to late at night and unleashes a pack of hound-like creatures to speed things up. They kill Thresh and force Katniss and Peeta to flee to the roof of the Cornucopia, where they encounter Cato. After a battle, Katniss wounds Cato with an arrow and Peeta hurls him to the creatures below. Katniss shoots Cato to spare him a prolonged death. With Peeta and Katniss apparently victorious, the rule change allowing two winners is suddenly revoked. Peeta tells Katniss to shoot him. Instead, she gives him half of the nightlock. However, before they can commit suicide, they are hastily proclaimed the victors of the 74th Hunger Games. Haymitch warns Katniss that she has made powerful enemies after her display of defiance. She and Peeta return to District 12, while Crane is locked in a room with a bowl of nightlock berries, and President Snow considers the situation.'''
# maker = eventMaker(line)
# maker.getEvent()
# for e in maker.events:
# 	print(e)
# with open('output.txt', 'w') as f:
# 	for event in maker.events:
# 		sentence = " ".join(event)
# 		f.write(sentence)
# 		f.write('\n')
# quit()
	#print(sentence+"@@@@"+line)

