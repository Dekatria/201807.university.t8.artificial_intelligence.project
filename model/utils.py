
from os.path import isfile, join
import json
import numpy as np
import pickle
import h5py

def load_image_features(data_dir, split):
	import h5py
	img_features = None
	image_id_list = None
	with h5py.File( join( data_dir, (split + "_img_embed.h5")),"r") as hf:
		img_features = np.array(hf.get("img_features"))
	with h5py.File( join( data_dir, (split + "_image_id_list.h5")),"r") as hf:
		image_id_list = np.array(hf.get("image_id_list"))
	return img_features, image_id_list

def load_questions_answers(data_dir = "../data"):
	qa_data_file = join(data_dir, "qa_data_file.pkl")
	
	if isfile(qa_data_file):
		with open(qa_data_file, "rb") as f:
			data = pickle.load(f)
			return data