import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
from math import pi, atan, tan, sin, cos, sqrt
from math import degrees as deg
from math import radians as rad
import os
from MeshPly import MeshPly
from pytless import inout,renderer
import json

bridge = CvBridge()
config_file = open("config.json", 'r')
config = json.load(config_file)

data_path = config['GENERATOR_CONFIG']['DATA_PATH']
save_path = config['GENERATOR_CONFIG']['SAVE_PATH']


objects = os.listdir(data_path)
num_objects = len(objects)

camera_parameters = config['GENERATOR_CONFIG']['CAMERA_INTRINSIC']
K = np.zeros((3, 3), dtype='float64')
K[0, 0], K[0, 2] = camera_parameters[0], camera_parameters[2]
K[1, 1], K[1, 2] = camera_parameters[4], camera_parameters[5]
K[2, 2] = 1.



for obj in objects:
	save_count = 0

	if not os.path.isdir("%s/%s"%(save_path, obj)):
		os.makedirs("%s/%s"%(save_path, obj))
	if not os.path.isdir("%s/%s/mask"%(save_path, obj)):
		os.makedirs("%s/%s/mask"%(save_path, obj))
	if not os.path.isdir("%s/%s/labels"%(save_path, obj)):
		os.makedirs("%s/%s/labels"%(save_path, obj))

	meshname  = "%s/%s.ply"%(config['GENERATOR_CONFIG']['MODEL_PATH'],obj)
	mesh               = MeshPly(meshname) 
	vertices           = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
	vertices = vertices* 1000.0
	min_x = np.min(vertices[0,:])
	max_x = np.max(vertices[0,:])
	min_y = np.min(vertices[1,:])
	max_y = np.max(vertices[1,:])
	min_z = np.min(vertices[2,:])
	max_z = np.max(vertices[2,:])

	corners = np.array([[min_x, min_y, min_z],
						[min_x, min_y, max_z],
						[min_x, max_y, min_z],
						[min_x, max_y, max_z],
						[max_x, min_y, min_z],
						[max_x, min_y, max_z],
						[max_x, max_y, min_z],
						[max_x, max_y, max_z]])

	corners = np.concatenate((np.transpose(corners), np.ones((1,8))), axis=0)
	corners3D = corners
	
	model = inout.load_ply(meshname)
	model['pts'] = model['pts']*1000.0

	ins_list = os.listdir(data_path + "/%s/inspection"%obj)
	ins_list.sort()

	img_count = 0
	for ins_name in ins_list:
		ins_num = int(ins_name[3:-4])

		tmp_bgr = cv2.imread(data_path + "/%s/JPEGImages/%.6i.png"%(obj,ins_num), cv2.IMREAD_GRAYSCALE)
		rotation = np.load(data_path + "/%s/rotation/%.6i.npy"%(obj,ins_num))
		translation = np.load(data_path + "/%s/translation/%.6i.npy"%(obj,ins_num))

		transformation = np.concatenate((rotation, translation), axis=1)
		projections_2d = np.zeros((2, corners3D.shape[1]), dtype='float32')
		camera_projection = (K.dot(transformation)).dot(corners3D)
		projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
		projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]

		proj_2d_gt = np.transpose(projections_2d)

		ori_img = tmp_bgr.copy()

		surf_color = (1,0,0)
		im_size = (ori_img.shape[1],ori_img.shape[0])
		ren_rgb = renderer.render(model, im_size, K, rotation, translation,
								  surf_color=surf_color, mode='rgb')
		ren_gray = cv2.cvtColor(ren_rgb, cv2.COLOR_RGB2GRAY)
		mask = np.zeros((ren_gray.shape[0], ren_gray.shape[1]))
		mask[ren_gray!=0]=255

		keypoint_list = []

		for i in range(8):
			for j in range(2):
				if j%2 == 0:
					tmp = proj_2d_gt[i][j] / 640
					keypoint_list.append(tmp)
				else:
					tmp = proj_2d_gt[i][j] / 480
					keypoint_list.append(tmp)
					
		y_list, x_list = mask.nonzero()
		x_min, x_max = x_list.min(), x_list.max()
		y_min, y_max = y_list.min(), y_list.max()
		center_x = (x_min + x_max) / 2
		center_y = (y_min + y_max) / 2
		width = x_max - x_min
		height = y_max - y_min

		width = width / float(640)
		center_x = center_x / float(640)
		height = height / float(480)
		center_y = center_y / float(480)
		class_id = 0

		f = open("%s/%s/labels/%.6i.txt"%(save_path, obj, ins_num),'w')
		f.write("%i %f %f "%(class_id, center_x, center_y))
		for i in range(16):
			f.write("%f "%keypoint_list[i])
		f.write("%f %f"%(width, height))
		f.close()

		cv2.imwrite("%s/%s/mask/%.4i.png"%(save_path, obj, ins_num),mask)

		print("%i/%i"%(img_count,len(ins_list)))
		img_count += 1