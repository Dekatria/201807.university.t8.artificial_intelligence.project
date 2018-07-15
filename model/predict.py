import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import vis_lstm_model
import numpy as np
from os.path import isfile, join
import utils
import time
import re

slim = tf.contrib.slim
resnet = nets.resnet_v2

tf.app.flags.DEFINE_string("image_path", "test.jpg", "directory of image")

tf.app.flags.DEFINE_string("img_checkpoint_path", "./data/pretrain/resnet152/resnet_v2_152.ckpt",
                           "directory of checkpoint files for image feature extraction")

tf.app.flags.DEFINE_string("checkpoint_path", "./data/pretrain/model",
                           "directory of checkpoint files for overall model")

tf.app.flags.DEFINE_integer("num_lstm_layers", 2, "number of lstm layers")

tf.app.flags.DEFINE_integer(
    "img_feat_len", 1001, "length of image feature vector")

tf.app.flags.DEFINE_integer("rnn_size", 300, "size of rnn")

tf.app.flags.DEFINE_integer(
    "que_feat_len", 300, "length of question feature vector")

tf.app.flags.DEFINE_integer("word_dropout", 0.5, "dropout rate of word nodes")

tf.app.flags.DEFINE_integer("img_dropout", 0.5, "dropout rate of image nodes")

tf.app.flags.DEFINE_string("data_dir", "./data", "directory of data")

tf.app.flags.DEFINE_string("question", "What colour are her eyes?", "question")

FLAGS = tf.app.flags.FLAGS


def main():

    print ("Image:", FLAGS.image_path)
    print ("Question:", FLAGS.question)

    vocab_data = utils.get_question_answer_vocab(FLAGS.data_dir)
    qvocab = vocab_data['question_vocab']
    q_map = {vocab_data['question_vocab'][qw]
        : qw for qw in vocab_data['question_vocab']}

    with tf.Graph().as_default():
        images = tf.placeholder("float32", [None, 224, 224, 3])
        with slim.arg_scope(resnet.resnet_arg_scope()):
            net, _ = resnet.resnet_v2_152(images, 1001, is_training=False)

        restorer = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	        start = time.clock()
	        image_array = utils.load_image_array(FLAGS.image_path)
	        image_feed = np.ndarray((1, 224, 224, 3))
	        image_feed[0:, :, :] = image_array

	        # checkpoint = tf.train.latest_checkpoint(FLAGS.img_checkpoint_path)
	        checkpoint = FLAGS.img_checkpoint_path
	        restorer.restore(sess, checkpoint)
	        print("Image Model loaded")
	        feed_dict = {images: image_feed}
	        img_feature = sess.run(net, feed_dict=feed_dict)
	        img_feature = np.squeeze(img_feature)
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

    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')
    question_ids = np.zeros(
        (1, vocab_data['max_question_length']), dtype='int32')
    question_words = re.findall(word_regex, FLAGS.question)
    base = vocab_data['max_question_length'] - len(question_words)
    for i in range(0, len(question_words)):
        if question_words[i] in question_vocab:
            question_ids[0][base + i] = question_vocab[question_words[i]]
        else:
            question_ids[0][base + i] = question_vocab['UNK']

    ans_map = {vocab_data['answer_vocab'][ans]
        : ans for ans in vocab_data['answer_vocab']}

    with tf.Graph().as_default():
        model = vis_lstm_model.Vis_lstm_model(model_options)
        input_tensors, t_prediction, t_ans_probab = model.build_generator()
        restorer = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
	        restorer.restore(sess, checkpoint)
	        pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
	            input_tensors['img']: np.reshape(img_feature, [1,1001]),
	            input_tensors['sentence']: question_ids,
	        })

    print("Ans:", ans_map[pred[0]])
    answer_probab_tuples = [(-answer_probab[0][idx], idx)
                            for idx in range(len(answer_probab[0]))]
    answer_probab_tuples.sort()
    print("Top Answers")
    for i in range(5):
        print(ans_map[answer_probab_tuples[i][1]])

if __name__ == '__main__':
    main()
