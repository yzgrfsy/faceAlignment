import menpo.io as mio
import menpo
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
            if os.path.splitext(file)[1] == '.jpg':
                L.append(os.path.splitext(file)[0])
    return L
image_path_pred = "/home/yuzhg/HH/46/46-3pre"
#加载保存的模型
fitter = mio.import_pickle('pretrained266-3-30_aam.pkl')()
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
    # points = [[39, 187], [70, 298], [53, 331], [64, 357], [91, 358], [128, 357], [161, 356], [172, 327], [151, 296],
    #           [180, 191], [144, 73], [66, 75],
    #           [146, 196], [151, 184], [164, 184], [160, 194], [73, 193], [62, 198], [51, 190], [63, 182], [113, 231],
    #           # [131, 301], [101, 302]]
    # points = [[39, 187], [70, 298], [53, 331], [64, 357], [91, 358], [128, 357], [161, 356] ]
    # points = [[39, 187], [70, 298], [53, 331]]

    # points = [[77,252], [170,248], [127,300]]
    # for 46cows points
    points = [[24,213], [219,209], [134,488]]
    # points = [[28,218], [208,218], [118, 482]]
    # points = [[25,216], [210,216], [138, 499]]
    # points = [[17,82], [80,82], [50,159]]
    # points = [[10,73], [85,73], [50,153]]
    # for 266cows points
    # points = [[15,73], [80,73], [50,153]]
    initial_shape = menpo.shape.PointCloud(points, copy=True)
    # print("initial_shape:",initial_shape)
    result = fitter.fit_from_shape(i, initial_shape, max_iters=30, gt_shape=None)

    # print result
    # print('result:',result)
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