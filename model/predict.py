import tensorflow as tf
import vis_lstm_model
import argparse
import numpy as np
from os.path import isfile, join
import utils
import re

tf.app.flags.DEFINE_string("image_path", None, "directory of image")

tf.app.flags.DEFINE_string("checkpoint_path", "./data/pretrain/model", "directory of checkpoint files")

tf.app.flags.DEFINE_integer("num_lstm_layers", 2, "number of lstm layers")

tf.app.flags.DEFINE_integer("img_feat_len", 1001, "length of image feature vector")

tf.app.flags.DEFINE_integer("rnn_size", 300, "size of rnn")

tf.app.flags.DEFINE_integer("que_feat_len", 300, "length of question feature vector")

tf.app.flags.DEFINE_integer("word_dropout", 0.5, "dropout rate of word nodes")

tf.app.flags.DEFINE_integer("img_dropout", 0.5, "dropout rate of image nodes")

tf.app.flags.DEFINE_string("data_dir", "./data", "directory of data")

tf.app.flags.DEFINE_string("question", None, "question")

FLAGS = tf.app.flags.FLAGS

def main():

	print "Image:", FLAGS.image_path
	print "Question:", FLAGS.question

	vocab_data = utils.get_question_answer_vocab(FLAGS.data_dir)
	qvocab = vocab_data['question_vocab']
	q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}
	
	images = tf.placeholder("float32", [None, 224, 224, 3])
	image_array = load_image_array(image_path)
	image_feed = np.ndarray((1,224,224,3))
	image_feed[0:,:,:] = image_array
	feed_dict  = { images : image_feed }
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_features = sess.run(fc7_tensor, feed_dict = feed_dict)
	sess.close()
	return fc7_features

	fc7_features = utils.extract_fc7_features(args.image_path, join(args.data_dir, 'vgg16.tfmodel'))
	
	model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'fc7_feature_length' : args.fc7_feature_length,
		'lstm_steps' : vocab_data['max_question_length'] + 1,
		'q_vocab_size' : len(vocab_data['question_vocab']),
		'ans_vocab_size' : len(vocab_data['answer_vocab'])
	}
	
	question_vocab = vocab_data['question_vocab']
	word_regex = re.compile(r'\w+')
	question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
	question_words = re.findall(word_regex, args.question)
	base = vocab_data['max_question_length'] - len(question_words)
	for i in range(0, len(question_words)):
		if question_words[i] in question_vocab:
			question_ids[0][base + i] = question_vocab[ question_words[i] ]
		else:
			question_ids[0][base + i] = question_vocab['UNK']

	ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_prediction, t_ans_probab = model.build_generator()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)
	
	pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
        input_tensors['fc7']:fc7_features,
        input_tensors['sentence']:question_ids,
    })

	
	print "Ans:", ans_map[pred[0]]
	answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
	answer_probab_tuples.sort()
	print "Top Answers"
	for i in range(5):
		print ans_map[ answer_probab_tuples[i][1] ]

if __name__ == '__main__':
	main()