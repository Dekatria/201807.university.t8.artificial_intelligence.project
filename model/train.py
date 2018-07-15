import tensorflow as tf
import vis_lstm_model
import utils
import numpy as np

tf.app.flags.DEFINE_integer("num_lstm_layers", 2, "number of lstm layers")

tf.app.flags.DEFINE_integer("img_feat_len", 1001, "length of image feature vector")

tf.app.flags.DEFINE_integer("rnn_size", 300, "size of rnn")

tf.app.flags.DEFINE_integer("que_feat_len", 300, "length of question feature vector")

tf.app.flags.DEFINE_integer("word_dropout", 0.5, "dropout rate of word nodes")

tf.app.flags.DEFINE_integer("img_dropout", 0.5, "dropout rate of image nodes")

tf.app.flags.DEFINE_string("data_dir", "./data", "directory of data")

tf.app.flags.DEFINE_integer("batch_size", 200, "size of batches")

tf.app.flags.DEFINE_integer("learning_rate", 0.001, "learning rate")

tf.app.flags.DEFINE_integer("epochs", 200, "number of epochs")

tf.app.flags.DEFINE_string("checkpoint_path", None, "directory of checkpoint files")

tf.app.flags.DEFINE_bool("debug", True, "debug subroutine")

FLAGS = tf.app.flags.FLAGS

def main():

	print("Reading QA DATA")
	qa_data = utils.load_questions_answers(FLAGS.data_dir)                                                           
	
	print("Reading image features")
	img_features, image_id_list = utils.load_image_features(FLAGS.data_dir, "train")
	print("img features", img_features.shape)
	print("image_id_list", image_id_list.shape)

	image_id_map = {}
	for i in range(len(image_id_list)):
		image_id_map[ image_id_list[i] ] = i
	
	ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

	model_options = {
		"num_lstm_layers" : FLAGS.num_lstm_layers,
		"rnn_size" : FLAGS.rnn_size,
		"embedding_size" : FLAGS.que_feat_len,
		"word_emb_dropout" : FLAGS.word_dropout,
		"image_dropout" : FLAGS.img_dropout,
		"img_feature_length" : FLAGS.img_feat_len,
		"lstm_steps" : qa_data["max_question_length"] + 1,
		"q_vocab_size" : len(qa_data["question_vocab"]),
		"ans_vocab_size" : len(qa_data["answer_vocab"])
	}
	
	
	
	model = vis_lstm_model.Vis_lstm_model(model_options)
	input_tensors, t_loss, t_accuracy, t_p = model.build_model()
	train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(t_loss)
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	
	saver = tf.train.Saver()
	if FLAGS.checkpoint_path:
		saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_path))

	for i in range(FLAGS.epochs):
		batch_no = 0

		while (batch_no*FLAGS.batch_size) < len(qa_data["training"]):
			sentence, answer, img = get_training_batch(batch_no, FLAGS.batch_size, img_features, image_id_map, qa_data, "train")
			_, loss_value, accuracy, pred = sess.run([train_op, t_loss, t_accuracy, t_p], 
				feed_dict={
					input_tensors["img"]:img,
					input_tensors["sentence"]:sentence,
					input_tensors["answer"]:answer
				}
			)
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
			
		save_path = saver.save(sess, "./data/pretrain/model/model{}.ckpt".format(i))

def get_training_batch(batch_no, batch_size, img_features, image_id_map, qa_data, split):
	qa = None
	if split == "train":
		qa = qa_data["training"]
	else:
		qa = qa_data["validation"]

	si = (batch_no * batch_size)%len(qa)
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