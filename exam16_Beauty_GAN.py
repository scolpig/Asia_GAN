import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def align_faces(img, detector, sp):

img_result = img.copy()
dets = detector(img)
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1,figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h,linewidth=2, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3,
                                edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
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

