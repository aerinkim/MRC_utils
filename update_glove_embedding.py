
import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import pickle
import spacy
import tqdm
import numpy as np
from os.path import basename, dirname
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from my_utils.utils import set_environment
from my_utils.tokenizer import reform_text, Vocabulary, END
from my_utils.log_wrapper import create_logger
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
from my_utils.squad_eval_v2 import my_evaluation




# Get the list of the train vocab only


with open('data_v2/squad_meta_v2_train_only.pick', 'rb') as f:
	train_only_meta = pickle.load(f)

with open('data_v2/squad_meta_v2_train_and_dev.pick', 'rb') as f:
	train_and_dev_meta = pickle.load(f)


train_only_vocab=train_only_meta['vocab']
train_only_vocab_list = train_only_vocab.get_vocab_list()
len(train_only_vocab.get_vocab_list())

train_and_dev_vocab=train_and_dev_meta['vocab']
train_and_dev_vocab_list = train_and_dev_vocab.get_vocab_list()
len(train_and_dev_vocab.get_vocab_list())


# get the index of train_only_meta's vocab in train_and_dev_meta

index_to_update=[]
for word in train_only_vocab_list:
	try:
		index_to_update.append(train_and_dev_vocab_list.index(word))
	except:
		print (word)
		pass

train_only_vocab_list.remove('Lovin')

# using that index, get the embedding of bestcheckpoint.pt


model_path = 'v2_FGSM_max10_original_200_25.pt'
checkpoint = torch.load(model_path)
checkpoint['state_dict']['network']['lexicon_encoder.embedding.weight'].shape
trained_embedding=checkpoint['state_dict']['network']['lexicon_encoder.embedding.weight']


# Update the Glove embedding.


with open("glove.840B.300d.updated.txt", mode="w" , encoding="utf8") as outfile: 
	with open('data_v2/glove.840B.300d.txt', encoding="utf8") as f:
		glove_token_sequence = []
		emb = np.zeros((2196017, 300))
		i=0
		for line in f:
			elems = line.split()
			token = ' '.join(elems[0:-300])
			glove_token_sequence.append(token)
			emb[i] = [float(v) for v in elems[-300:]]
			i = i+1

		print('sanity check. embedding shape : ', emb)
		print('sanity check. embedding shape : ', emb.shape)

		for token in train_only_vocab_list:
			try:
				glove_index = glove_token_sequence.index(token)
				# Get the index of "train_and_dev_vocab_list"
				"""
				In [33]: len(train_and_dev_vocab_list)
				Out[33]: 89679

				In [34]: len(trained_embedding)
				Out[34]: 89679
				"""
				row_idx = train_and_dev_vocab_list.index(token)
				emb[glove_index] = trained_embedding[row_idx]
			except Exception as e:
				print (e)
				pass

		emb =  np.round(emb,6)
		for i in range(0,len(glove_token_sequence)):
			token = glove_token_sequence[i]
			dim_300 = ' '.join(str(j) for j in emb[i]) 
			outfile.write("%s %s\n" %(token, dim_300))
			if i % 100000 == 0:
				print ("write file:%s %s %s\n" %(str(i), token, dim_300))

