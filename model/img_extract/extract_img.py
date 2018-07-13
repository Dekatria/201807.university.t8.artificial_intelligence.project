import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import json
from os.path import join
import numpy as np
import time
import h5py
import utils

slim = tf.contrib.slim
resnet = nets.resnet_v2

train_log_dir = "../log"

tf.app.flags.DEFINE_string("split", "train", "partition usage of dataset")

tf.app.flags.DEFINE_string("data_dir", "../data", "directory of dataset")

tf.app.flags.DEFINE_integer("output_size", 1001, "size of output of embeding")

tf.app.flags.DEFINE_string(
    "checkpoint_path", "pretrain/resnet152/resnet_v2_152.ckpt", "directory of checkpoint file")

tf.app.flags.DEFINE_integer("batch_size", 10, "size of batches")

FLAGS = tf.app.flags.FLAGS


def main():

	with tf.Graph().as_default():

		image_ids = {}
		if FLAGS.split == "train":
			filename = join(
	            FLAGS.data_dir, 'annotations/v2_OpenEnded_mscoco_train2014_questions.json')
		else:
			filename = join(
				FLAGS.data_dir, 'annotations/v2_OpenEnded_mscoco_val2014_questions.json')
		with open(filename) as f:
			raw = json.loads(f.read())
		for que in raw["questions"]:
			image_ids[que["image_id"]] = 1

		image_id_list = [img_id for img_id in image_ids]
		print("Total Images", len(image_id_list))

		images = tf.placeholder("float", [None, 224, 224, 3])

		with slim.arg_scope(resnet.resnet_arg_scope()):
			net, _ = resnet.resnet_v2_152(images, FLAGS.output_size, is_training=False)

		restorer = tf.train.Saver()

		optimizer = tf.train.AdamOptimizer(learning_rate=.001)

		results = np.ndarray((len(image_id_list), FLAGS.output_size))
		idx = 0

		with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
			while idx < len(image_id_list):
				start = time.clock()
				image_batch = np.ndarray( (FLAGS.batch_size, 224, 224, 3 ) )

				count = 0
				for i in range(0, FLAGS.batch_size):
					if idx >= len(image_id_list):
						break
					image_file = join(FLAGS.data_dir, '%s2017/%.12d.jpg' %
	                              (FLAGS.split, image_id_list[idx]))
					image_batch[i,:,:,:] = utils.load_image_array(image_file)
					idx += 1
					count += 1
				
				
				feed_dict  = { images : image_batch[0:count,:,:,:] }
				
				checkpoint = join(FLAGS.data_dir, FLAGS.checkpoint_path)
				restorer.restore(sess, checkpoint)
				print("Model Restored")
				pred_batch = sess.run(net, feed_dict=feed_dict)
				# print(np.squeeze(pred_batch).shape)
				results[(idx - count):idx, :] = np.squeeze(pred_batch)[0:count,:]
				end = time.clock()
				print("Time for batch 10 photos", end - start)
				print("Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/10.0)

				print("Images Processed", idx)

		print("Saving image features")

		h5f_img_embed = h5py.File(
			join(FLAGS.data_dir, FLAGS.split + 'img_embed.h5'), 'w')
		h5f_img_embed.create_dataset('img_features', data=results)
		h5f_img_embed.close()

		print("Saving image id list")
		h5f_image_id_list = h5py.File(
			join(FLAGS.data_dir, FLAGS.split + '_image_id_list.h5'), 'w')
		h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
		h5f_image_id_list.close()
		print("Done!")

if __name__ == '__main__':
    main()


