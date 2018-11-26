# # -*- coding:utf-8 -*-

import time,json
import os
write_data = []
###read data###
def file_name_except_format(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.splitext(file)[0])
    return L
def tansfer_single(filename,source_path,transf_path):
    with open(os.path.join(source_path,filename+'.json'), 'r',encoding='utf-8') as load_f:
        print(filename)
        write_data = json.load(load_f)
        list_points = []
        # print (write_data["shapes"]["points"])
        for key, value in write_data.items():
            # print("key:%s, value:%s" % (key, value))
            try:
                if key=="shapes" and len(value)>0:
                    for item in value:
                        for key, value in item.items():
                            if key=="points":
                                print("key:%s, value:%s" % (key, value))
                                for item in value:
                                    print(item)
                                    list_points.append(item)
            except Exception:
                print(Exception)
        print(list_points)
        # openFileHandle = open(file, 'r')writeFileHandle=open('Temp','w')
        file_copy=os.path.join(transf_path,filename+'.pts')
        with open(file_copy,'w',encoding='utf-8') as writeFileHandle:
            writeFileHandle.write("version: 1"+'\n')
            writeFileHandle.write("n_points: "+str(len(list_points))+'\n')
            writeFileHandle.write("{"+'\n')
            for i in range(len(list_points)):
                writeFileHandle.write(str(list_points[i][0])+" "+str(list_points[i][1])+'\n')
            writeFileHandle.write("}"+'\n')
            writeFileHandle.close()
#source_path指json格式注释文件夹地址，transf_path指转换后的pts注释文件夹地址
def main():
    source_path=r"D:\电信研究院\人脸矫正\labelme-master\examples\tutorial\test"
    transf_path=r"D:\电信研究院\人脸矫正\labelme-master\examples\transfer\test"
    file_list = file_name_except_format(source_path)
    for i in file_list:
        tansfer_single(i,source_path,transf_path)
main()

