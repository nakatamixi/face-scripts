# coding:utf-8

import cv2
import numpy as np
import sys, os
import re
 
def get_files():
	input_dir_or_file = sys.argv[1]
	if os.path.isfile(input_dir_or_file):
		return [input_dir_or_file]

	files = os.listdir(input_dir_or_file)
	return [input_dir_or_file + f for f in files]

def replace_file_path(file_path: str):
    file_path_without_ex = file_path.split(".")[:-1]
    return "_".join(file_path_without_ex).replace("/", "_")

files= get_files()
cascade = cv2.CascadeClassifier("./data/haarcascades/haarcascade_frontalface_alt.xml")
for f in files:
	image_gs = cv2.imread(f)
	face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))
	for i, rect in enumerate(face_list):
		x = rect[0]
		y = rect[1]
		width = rect[2]
		height = rect[3]
		dst = cv2.resize(image_gs[y:y + height, x:x + width], (200, 200), interpolation=cv2.INTER_LINEAR)
		save_path = './output/' + replace_file_path(str(f))+ str(i) + '.jpg'
		a = cv2.imwrite(save_path, dst)
