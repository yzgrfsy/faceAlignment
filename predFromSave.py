import menpo.io as mio
from menpo.visualize import print_progress
from menpofit.aam import HolisticAAM
from menpo.feature import igo
from pathlib import Path
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpofit.fitter import noisy_shape_from_bounding_box
from menpodetect import load_dlib_frontal_face_detector
import os
from menpofit.io import PickleWrappedFitter, image_greyscale_crop_preprocess
from functools import partial
from menpo.shape import PointDirectedGraph
#预测
def file_name_except_format(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                L.append(os.path.splitext(file)[0])
    return L
image_path_pred = "D:/电信研究院/人脸矫正/labelme-master/examples/transfer/pred"
#加载保存的模型
fitter = mio.import_pickle('pretrained12131_aam.pkl')()
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
    # Load detector
    detect = load_dlib_frontal_face_detector()
    # Detect
    bboxes = detect(i)
    print("{} detected faces.".format(len(bboxes)))
    # initial bbox
    # initial_bbox = bboxes[0]
    import numpy as np
    imHei=i.height
    imWid=i.width
    adjacency_matrix = np.array([[0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1],
                                 [1, 0, 0, 0]])
    points = np.array([[0, 0], [imWid, 0], [imWid, imHei], [0, imHei]])
    graph = PointDirectedGraph(points, adjacency_matrix)
    # fit image
    # result = fitter.fit_from_bb(i, initial_bbox, max_iters=[15, 5],
    #                             gt_shape=None)
    result = fitter.fit_from_bb(i, graph, max_iters=[20, 10],
                                gt_shape=None)
    # print result
    print('result',result)
    points_list=result.final_shape.points
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