import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim
import vis_lstm_model
import numpy as np
from os.path import isfile, join
import utils
import time
import re

def main():

	tf.app.flags.DEFINE_integer("num_lstm_layers", 2, "number of lstm layers")
	tf.app.flags.DEFINE_integer("img_feat_len", 1001, "length of image feature vector")
	tf.app.flags.DEFINE_integer("rnn_size", 300, "size of rnn")
	tf.app.flags.DEFINE_integer("que_feat_len", 300, "length of question feature vector")
	tf.app.flags.DEFINE_float("word_dropout", 0.5, "dropout rate of word nodes")
	tf.app.flags.DEFINE_float("img_dropout", 0.5, "dropout rate of image nodes")
	tf.app.flags.DEFINE_string("data_dir", "./data", "directory of data")
	tf.app.flags.DEFINE_integer("batch_size", 200, "size of batches")
	tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
	tf.app.flags.DEFINE_integer("epochs", 200, "number of epochs")
	tf.app.flags.DEFINE_string("checkpoint_path", None, "directory of checkpoint files")
	tf.app.flags.DEFINE_bool("debug", True, "debug subroutine")
	tf.app.flags.DEFINE_string("image_path","./data/val2014","image path")
	FLAGS = tf.app.flags.FLAGS
	
	print("Reading QA DATA")
	qa_data = utils.load_questions_answers(FLAGS.data_dir)                                                           
	vocab_data = utils.get_question_answer_vocab(FLAGS.data_dir)
	print("Reading image features")
	img_features, image_id_list = utils.load_image_features(FLAGS.data_dir, "validation")
	print("img features", img_features.shape)
	print("image_id_list", image_id_list.shape)
	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}
	resnet = nets.resnet_v2
	image_id_map = {}
	for i in range(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i
	
	
	with tf.Graph().as_default():
		images = tf.placeholder("float32", [None, 224, 224, 3])	
		with slim.arg_scope(resnet.resnet_arg_scope()):
			net, _ = resnet.resnet_v2_152(images, 1001, is_training=False)

		restorer = tf.train.Saver()

		with tf.Session() as sess:
			start = time.clock()
			
			image_array = utils.load_image_array(FLAGS.image_path)
			image_feed = np.ndarray((1, 224, 224, 3))
			image_feed[0:, :, :] = image_array

			checkpoint = FLAGS.checkpoint_path
			restorer.restore(sess, checkpoint)
			print("Image Model loaded")
			feed_dict = {images: image_feed}
			img_feature = sess.run(net, feed_dict=feed_dict)
			img_feature = np.squeeze(img_feature)
			#img_features2[] = img_feature
			end = time.clock()
			print("Time elapsed", end - start)
			print("Image processed")

	model_options = {
		'num_lstm_layers': FLAGS.num_lstm_layers,
		'rnn_size': FLAGS.rnn_size,
		'embedding_size': FLAGS.que_feat_len,
		'word_emb_dropout': FLAGS.word_dropout,
		'image_dropout': FLAGS.img_dropout,
		'img_feature_length': FLAGS.img_feat_len,
		'lstm_steps': vocab_data['max_question_length'] + 1,
		'q_vocab_size': len(vocab_data['question_vocab']),
		'ans_vocab_size': len(vocab_data['answer_vocab'])
	}
	
	# load cached model
	with tf.Graph().as_default():
		model = vis_lstm_model.Vis_lstm_model(model_options)
		input_tensors, t_prediction, t_ans_probab = model.build_generator()
		restorer = tf.train.Saver()
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
		restorer.restore(sess, checkpoint)
			
		batch_no = 0
		
		while (batch_no*FLAGS.batch_size) < len(qa_data["validation"]):
			sentence, answer, img = get_validation_batch(batch_no, FLAGS.batch_size, img_features, image_id_map, qa_data)
			
			_, loss_value, accuracy, pred = sess.run([t_prediction, t_ans_probab], feed_dict={
					input_tensors['img']: img,
					input_tensors["sentence"]:sentence,
					input_tensors['sentence']: answer,
					})
			
			batch_no += 1
			
			if FLAGS.debug:
				for idx, p in enumerate(pred):
					print(ans_map[p], ans_map[ np.argmax(answer[idx])])

				print("Loss", loss_value, batch_no, i)
				print("Accuracy", accuracy)
				print("---------------")
			else:
				print("Loss", loss_value, batch_no, i)
				print("Training Accuracy", accuracy)

def get_validation_batch(batch_no, batch_size, img_features, image_id_map, qa_data):
	qa = qa_data["validation"]
	
	si = (batch_no * batch_size)
	ei = min(len(qa), si + batch_size)
	n = ei - si
	sentence = np.ndarray( (n, qa_data["max_question_length"]), dtype = "int32")
	answer = np.zeros( (n, len(qa_data["answer_vocab"])))
	img = np.ndarray( (n,1001) )

	count = 0
	for i in range(si, ei):
		sentence[count,:] = qa[i]["question"][:]
		answer[count, qa[i]["answer"]] = 1.0
		img_index = image_id_map[ qa[i]["image_id"] ]
		img[count,:] = img_features[img_index][:]
		count += 1
	
	return sentence, answer, img

if __name__ == "__main__":
    main()