import tensorflow as tf
import numpy as np
import time
import resnet
import skimage.io
import synset

PATH = "./"
META_FN = PATH + "ResNet-L50.meta"
CHECKPOINT_FN = PATH + "ResNet-L50.ckpt"

IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img



if __name__ == "__main__":
    sess = tf.Session()

    batch_size = 1 
    num_classes = 1000

    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

    logits = resnet.inference(x, is_training = False, num_classes = num_classes, num_blocks = [3,4,6,3])
    logits = tf.nn.softmax(logits)

    img = load_image("data/waterbottle.jpg")
    img = img[:,:,[2,1,0]].reshape((1,224,224,3)) * 255
    img -= np.array(IMAGENET_MEAN_BGR)

    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, CHECKPOINT_FN)

    pred = sess.run(logits, {x:img})
    top_k = 5
    top = pred.ravel().argsort()[-top_k:][::-1]
    for t in top.ravel().tolist():
        for s in synset.synset_map.values():
            index = s["index"]
            if index == t:
                print ("%s %.3f%%" % (s["desc"], pred.ravel()[t] * 100))
                break
