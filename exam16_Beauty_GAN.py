"""
화장 안 한 이미지에 화장 한 이미지를 학습시키기
"""

import dlib #이미지처리 라이브러리 , 페이스 디텍션, 랜드마크 디텍션, 페이스 얼라이브
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf  #뷰티 gan은 텐서버전1에서 만들어서 1로 변환해준다
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()  #이미지에서 얼굴 영역 인식 모델 로드

# 얼굴의 랜드마크 점의 위치를 찾아주는 모델을 받아 불러오기(얼굴에서 점 5개 찾아주는 모델)
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')

#위에서 수행한 작업을 aligh_faces() 함수로 만들어 준다
def align_faces(img, detector, sp):  #원본이미지를 넣으면 align 완료된 얼굴이미지 반환하는 함수
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
    return faces

# 학습된 BeautyGAN 모델 불러오기
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph("./models/model.meta")
saver.restore(sess, tf.train.latest_checkpoint("./models"))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0") #source
Y = graph.get_tensor_by_name("Y:0") #source
Xs = graph.get_tensor_by_name("generator/xs:0") #output

#전처리 및 후처리 함수(Preprocess and Postprocess Functions)
def preprocess(img):
    return (img / 255.0 - 0.5) * 2  #-1~1 사이가 되도록 스케일링(0 ~ 255 -> -1 ~ 1)


def deprocess(img):
    return (img + 1) / 2   #원래 값으로 복원시킬 때 사용하는 함수, -1 ~ 1 -> 0 ~ 255

#이미지 다운로드
img1 = dlib.load_rgb_image("./imgs/NA.jpg")  #화장 안한 사진 가져오기
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image("./imgs/makeup/XMY-136.png")   #화장 한 사진 가져오기
img2_faces = align_faces(img2, detector, sp)

#plt로 나타내기
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

#실행(Run) : x: source (no-makeup) + y:reference (makeup) = xs: output
src_img = img1_faces[0] #소스 이미지
ref_img = img2_faces[0] #레퍼런스 이미지

X_img = preprocess(src_img)  #화장 안한 이미지를 스케일링
X_img = np.expand_dims(X_img, axis=0)  # #np.expand_dims() : 배열에 차원을 추가한다. 즉, (256,256,2) -> (1,256,256,3)

Y_img = preprocess(ref_img)   #화장 한 이미지를 스케일링
Y_img = np.expand_dims(Y_img, axis=0) #텐서플로에서 0번 axis는 배치 방향

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})  #i사진에 2를 적용시켜서 만들어내기
output_img = deprocess(output[0])

#plt그리기
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()











