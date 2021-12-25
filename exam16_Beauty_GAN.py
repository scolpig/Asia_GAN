import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def align_faces(img, detector, sp):

    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

# test_img = dlib.load_rgb_image('./imgs/17.jpg')
# test_faces = align_faces(test_img, detector, sp)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
# axes[0].imshow(test_img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 초기화?
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

def preprocess(img):
    return (img / 255. - 0.5) * 2
def deprocess(img):
    return (img + 1) / 2

img1 = dlib.load_rgb_image('./imgs/12.jpg')
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image('./imgs/makeup/XMY-266.png')
img2_faces = align_faces(img2, detector, sp)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

# 화장을 입히기
src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img}) # predict
output_img = deprocess(output[0])

fig, axes = plt.subplots(1,3,figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()

