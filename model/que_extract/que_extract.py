import tensorflow as tf 
import numpy as np
import collections
import random
from scipy import spatial

embed_dict_path = "../../data/glove.6B/glove.6B.300d.txt"
embed_vocab = []
embed = []
embed_dict = {}

with open(embed_dict_path, "r") as file:
	for line in file.readlines():
		row = line.strip().split()
		word = row[0]
		embed_vocab.append(word)
		embed_vec = [float(i) for i in row[1:]]
		embed_dict[word] = embed_vec

print("embeding dictionary loaded")
embed_vocab_size = len(embed_vocab)
embed_dim = len(embed_vec)

def prepare_text(text):
	data = text.split()
	data = np.reshape(np.array(data), [-1, ])
	return data

def build_dict(text):
	raw = []
	for row in text:
		raw += row
	count = collections.Counter(raw).most_common()
	for_dict = {}
	rev_dict = {}
	for word, _ in count:
		for_dict[word] = len(for_dict)
	for key, val in for_dict.items():
		rev_dict[val] = key
	return for_dict, rev_dict

def get_embed(word_dict):
	text_vocab = len(word_dict)
	text_vocab_list = sorted(word_dict.items(), key = lambda x: x[1])
	embed_arr = []

	for i in range(text_vocab):
		word = text_vocab_list[i][0]
		if word in embed_dict:
			embed_arr.append(embed_dict[word])
		else:
			embed_arr.append(np.random.uniform(-.2, .2, size=embed_dim))

	embed_arr = np.asarray(embed_arr)
	tree = spatial.KDTree(embed_arr)
	return embed_arr, tree

learning_rate = 0.001
n_input = len(setence)
n_hidden = 512

x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, embed_dim])

V