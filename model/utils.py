
from os.path import isfile, join
import json
import numpy as np
import pickle
import h5py
from scipy import misc

def load_image_features(data_dir, split):
	import h5py
	img_features = None
	image_id_list = None
	with h5py.File( join( data_dir, (split + "_img_embed.h5")),"r") as hf:
		img_features = np.array(hf.get("img_features"))
	with h5py.File( join( data_dir, (split + "_image_id_list.h5")),"r") as hf:
		image_id_list = np.array(hf.get("image_id_list"))
	return img_features, image_id_list

def load_questions_answers(data_dir = "./data"):
	qa_data_file = join(data_dir, "qa_data_file.pkl")
	
	if isfile(qa_data_file):
		with open(qa_data_file, "rb") as f:
			data = pickle.load(f)
			return data

def get_question_answer_vocab(data_dir = './data'):
	vocab_file = join(data_dir, 'vocab_file.pkl')
	vocab_data = pickle.load(open(vocab_file, "rb"))
	return vocab_data

def load_image_array(image_file):
	img = misc.imread(image_file)
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = misc.imresize(img, (224, 224))
	return (img_resized/255.0).astype('float32')