import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.aam import HolisticAAM
from menpo.feature import igo
from pathlib import Path
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpofit.fitter import noisy_shape_from_bounding_box
from menpodetect import load_dlib_frontal_face_detector
import os
def file_name_except_format(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                L.append(os.path.splitext(file)[0])
    return L
def process(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image
#训练集
path_to_images = 'D:/电信研究院/人脸矫正/labelme-master/examples/transfer'
training_images = []
for img in print_progress(mio.import_images(path_to_images, verbose=True)):
    training_images.append(process(img))
# %matplotlib inline，这步只能在jupyter里可视化
# from menpowidgets import visualize_images
# visualize_images(training_images)

aam = HolisticAAM(training_images, reference_shape=None,
                  diagonal=150, scales=(0.5, 1.0),
                  holistic_features=igo, verbose=True)

fitter = LucasKanadeAAMFitter(aam,
                              lk_algorithm_cls=WibergInverseCompositional,
                              n_shape=[3, 20], n_appearance=[30, 150])
# % matplotlib inline


# method to load a database
def load_database(path_to_images, crop_percentage, max_images=None):
    images = []
    # load landmarked images
    for i in mio.import_images(path_to_images, max_images=max_images, verbose=True):
        # crop image
        i = i.crop_to_landmarks_proportion(crop_percentage)

        # convert it to grayscale if needed
        if i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')

        # append it to the list
        images.append(i)
    return images


# 测试集
image_path_test = "D:/电信研究院/人脸矫正/labelme-master/examples/transfer/test"
test_images = load_database(image_path_test, 0.5, max_images=5)
fitting_results = []
for i in test_images:
    # obtain original landmarks
    gt_s = i.landmarks['PTS'].lms
    # generate perturbed landmarks
    s = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())
    # fit image
    fr = fitter.fit_from_shape(i, s, gt_shape=gt_s)
    fitting_results.append(fr)
    print('fr',type(fr),fr.final_shape.points,fr)
# from menpowidgets import visualize_fitting_result
#预测
image_path_pred = "D:/电信研究院/人脸矫正/labelme-master/examples/transfer/pred"

pred_images = []
# load landmarked images
for i in mio.import_images(image_path_pred, max_images=None, verbose=True):
    # convert it to grayscale if needed
    if i.n_channels == 3:
        i = i.as_greyscale(mode='luminosity')

    # append it to the list
    pred_images.append(i)
png_list=file_name_except_format(image_path_pred)
cnt=0
for i in pred_images:
    # obtain original landmarks
    # gt_s = i.landmarks['PTS'].lms
    # # generate perturbed landmarks
    # s = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())
    # # fit image
    # Load detector
    detect = load_dlib_frontal_face_detector()
    # Detect
    bboxes = detect(i)
    print("{} detected faces.".format(len(bboxes)))
    # initial bbox
    initial_bbox = bboxes[0]

    # fit image
    result = fitter.fit_from_bb(i, initial_bbox, max_iters=[15, 5],
                                gt_shape=None)

    # print result
    print('result',result)
    # fr = fitter.fit_from_shape(i, gt_shape=None)
    # fr = fitter.fit_from_shape(i, s, gt_shape=gt_s)
    points_list=result.final_shape.points
    # openFileHandle = open(file, 'r')writeFileHandle=open('Temp','w')
    file_copy = os.path.join(image_path_pred, png_list[cnt] + '.pts')
    with open(file_copy, 'w', encoding='utf-8') as writeFileHandle:
        writeFileHandle.write("version: 1" + '\n')
        writeFileHandle.write("n_points: " + str(len(points_list)) + '\n')
        writeFileHandle.write("{" + '\n')
        for i in range(len(points_list)):
            writeFileHandle.write(str(points_list[i][0]) + " " + str(points_list[i][1]) + '\n')
        writeFileHandle.write("}" + '\n')
        cnt=cnt+1
        writeFileHandle.close()