import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
# from PyQt5 import QtGui, QtCore
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.uic import loadUi
# import roslib
from math import pi, atan, tan, sin, cos, sqrt
from math import degrees as deg
from math import radians as rad
# import rospy
import os
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from geometry_msgs.msg import TransformStamped 
# from robotiq_85_msgs.msg import GripperCmd, GripperStat
# import urx
# from mask_rcnn_ros.msg import *
# from scipy import ndimage
# from singleshot import singleshot
# from apriltags2_ros.msg import AprilTagDetectionArray
from MeshPly import MeshPly
# from torchvision import datasets, transforms
# import PyKDL
from pytless import inout,renderer
# import pandas

bridge = CvBridge()

data_path = "/media/david/Samsung_T5/191001JL/diget_sand"
save_path = "/media/david/Samsung_T5/191001JL/generated"


objects = os.listdir(data_path)
num_objects = len(objects)

K = np.zeros((3, 3), dtype='float64')
# K[0, 0], K[0, 2] = 528.2528340378916, 317.6973305122578
# K[1, 1], K[1, 2] = 529.980463017893, 246.0399589788932
K[0, 0], K[0, 2] = 617.6207275390625, 322.30206298828125
K[1, 1], K[1, 2] = 617.60205078125, 245.4082794189453
K[2, 2] = 1.



for obj in objects:
	save_count = 0

	if not os.path.isdir("%s/%s"%(save_path, obj)):
		os.makedirs("%s/%s"%(save_path, obj))
	# if not os.path.isdir("%s/%s/JPEGImages"%(save_path, obj)):
	# 	os.makedirs("%s/%s/JPEGImages"%(save_path, obj))
	if not os.path.isdir("%s/%s/mask"%(save_path, obj)):
		os.makedirs("%s/%s/mask"%(save_path, obj))
	# if not os.path.isdir("%s/%s/changed_img"%(save_path, obj)):
	# 	os.makedirs("%s/%s/changed_img"%(save_path, obj))
	if not os.path.isdir("%s/%s/labels"%(save_path, obj)):
		os.makedirs("%s/%s/labels"%(save_path, obj))
	# if not os.path.isdir("%s/%s/depth"%(save_path, obj)):
	# 	os.makedirs("%s/%s/depth"%(save_path, obj))

	meshname  = "/media/david/Samsung_T5/191001JS/save/diget_sand/obj/%s.ply"%obj
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

		# box_3d_color = (255,255,255)

		# ################################################

		# x1 = int(proj_2d_gt[0][0])
		# y1 = int(proj_2d_gt[0][1])
		# x2 = int(proj_2d_gt[2][0])
		# y2 = int(proj_2d_gt[2][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[4][0])
		# y1 = int(proj_2d_gt[4][1])
		# x2 = int(proj_2d_gt[6][0])
		# y2 = int(proj_2d_gt[6][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[0][0])
		# y1 = int(proj_2d_gt[0][1])
		# x2 = int(proj_2d_gt[4][0])
		# y2 = int(proj_2d_gt[4][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[2][0])
		# y1 = int(proj_2d_gt[2][1])
		# x2 = int(proj_2d_gt[6][0])
		# y2 = int(proj_2d_gt[6][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# ################################################

		# x1 = int(proj_2d_gt[1][0])
		# y1 = int(proj_2d_gt[1][1])
		# x2 = int(proj_2d_gt[3][0])
		# y2 = int(proj_2d_gt[3][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[5][0])
		# y1 = int(proj_2d_gt[5][1])
		# x2 = int(proj_2d_gt[7][0])
		# y2 = int(proj_2d_gt[7][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[1][0])
		# y1 = int(proj_2d_gt[1][1])
		# x2 = int(proj_2d_gt[5][0])
		# y2 = int(proj_2d_gt[5][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[3][0])
		# y1 = int(proj_2d_gt[3][1])
		# x2 = int(proj_2d_gt[7][0])
		# y2 = int(proj_2d_gt[7][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# ################################################

		# x1 = int(proj_2d_gt[0][0])
		# y1 = int(proj_2d_gt[0][1])
		# x2 = int(proj_2d_gt[1][0])
		# y2 = int(proj_2d_gt[1][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[2][0])
		# y1 = int(proj_2d_gt[2][1])
		# x2 = int(proj_2d_gt[3][0])
		# y2 = int(proj_2d_gt[3][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[4][0])
		# y1 = int(proj_2d_gt[4][1])
		# x2 = int(proj_2d_gt[5][0])
		# y2 = int(proj_2d_gt[5][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[6][0])
		# y1 = int(proj_2d_gt[6][1])
		# x2 = int(proj_2d_gt[7][0])
		# y2 = int(proj_2d_gt[7][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# ################################################

		# x1 = int(proj_2d_gt[1][0])
		# y1 = int(proj_2d_gt[1][1])
		# x2 = int(proj_2d_gt[7][0])
		# y2 = int(proj_2d_gt[7][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[3][0])
		# y1 = int(proj_2d_gt[3][1])
		# x2 = int(proj_2d_gt[5][0])
		# y2 = int(proj_2d_gt[5][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# ################################################

		# x1 = int(proj_2d_gt[2][0])
		# y1 = int(proj_2d_gt[2][1])
		# x2 = int(proj_2d_gt[7][0])
		# y2 = int(proj_2d_gt[7][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# x1 = int(proj_2d_gt[3][0])
		# y1 = int(proj_2d_gt[3][1])
		# x2 = int(proj_2d_gt[6][0])
		# y2 = int(proj_2d_gt[6][1])
		# cv2.line(ori_img,(x1,y1),(x2,y2),box_3d_color,2)

		# depth = np.load(data_path + "/%s/depth/%s.npy"%(obj,img_name[0:6]))
		
		# label = pandas.read_csv(data_path + "/%s/labels/%s.txt"%(obj,img_name[0:6]), header=None, delim_whitespace=True)
		# label = label.values

		# depth_mask = np.zeros(depth.shape)
		# for i in range(depth.shape[0]):
		# 	for j in range(depth.shape[1]):
		# 		if depth[i,j] == 0 :
		# 			depth_mask[i,j] = 255
		# depth_mask = depth_mask.astype(np.uint8)
		# # depth_tmp = depth_tmp.astype(np.uint8)
		# dst = cv2.inpaint(depth ,depth_mask, 3, flags=cv2.INPAINT_TELEA)

		# surf_color = (0,1,0)
		# im_size = (tmp_bgr.shape[1],tmp_bgr.shape[0])
		# ren_rgb = renderer.render(model, im_size, K, rotation, translation,
		# 						  surf_color=surf_color, bg_color=(1.0, 1.0, 1.0, 0.0), mode='rgb')
		# ren_gray = cv2.cvtColor(ren_rgb, cv2.COLOR_RGB2GRAY)
		# mask = np.zeros((ren_gray.shape[0], ren_gray.shape[1]))
		# mask[ren_gray!=255]=255
		
		# vis_rgb = ren_rgb.astype(np.float)
		# vis_rgb = vis_rgb.astype(np.uint8)

		# result = vis_rgb
		# mask = mask

		# mask_nonzero = np.nonzero(mask)

		# y_length = np.max(mask_nonzero[0]) - np.min(mask_nonzero[0])
		# x_length = np.max(mask_nonzero[1]) - np.min(mask_nonzero[1])

		# y_center = (np.max(mask_nonzero[0]) + np.min(mask_nonzero[0]))/2
		# x_center = (np.max(mask_nonzero[1]) + np.min(mask_nonzero[1]))/2

		# # if y_length > x_length:
		# # 	ymin = y_center - y_length/2 - 50	
		# # 	ymax = y_center + y_length/2 + 50
		# # 	xmin = x_center - y_length/2 - 50
		# # 	xmax = x_center + y_length/2 + 50
		# # else:
		# # 	ymin = y_center - x_length/2 - 50
		# # 	ymax = y_center + x_length/2 + 50
		# # 	xmin = x_center - x_length/2 - 50
		# # 	xmax = x_center + x_length/2 + 50

		# ymin = y_center - y_length/2 - 10
		# ymax = y_center + y_length/2 + 10
		# xmin = x_center - x_length/2 - 10
		# xmax = x_center + x_length/2 + 10

		# ymin_save = ymin
		# xmin_save = xmin

		# result = result[ymin:ymax, xmin:xmax]
		# mask = mask[ymin:ymax, xmin:xmax]

		# tmp_save = tmp_bgr.copy()
		# tmp_save = tmp_save[ymin:ymax, xmin:xmax]
		# depth_save = dst[ymin:ymax, xmin:xmax]
		# new_depth = depth_save.copy()
		# depth_max = depth_save.max()
		# # new_depth[depth_save > depth_max - 20] = 0
		# # new_depth[mask==0] = 0
		
		# new_depth_max = new_depth[new_depth!=0].max()
		# # new_depth_std = new_depth[new_depth!=0].var()
		# # new_depth[new_depth==0] = new_depth_max
		# # cv2.imshow("img",new_depth*40)

		# print(new_depth.min(), new_depth.max())
		# # new_depth = cv2.normalize(new_depth, None, 0, 255, cv2.NORM_MINMAX)
		# new_depth = new_depth.astype(np.uint8)


		# f = open("%s/%s/labels/%.6i.txt"%(save_path, obj, save_count),'w')
		# f.write("%i %i %i %i %i %i %i %i"%(label[0][0]-xmin_save, label[0][1]-ymin_save, label[0][2]-xmin_save, label[0][3]-ymin_save, label[0][4]-xmin_save, label[0][5]-ymin_save, label[0][6]-xmin_save, label[0][7]-ymin_save))
		# f.close()

		# # cv2.imwrite("%s/%s/JPEGImages/%.6i.png"%(save_path, obj, save_count),tmp_save)
		# cv2.imwrite("%s/%s/mask/%.4i.png"%(save_path,obj, save_count),mask)
		# # cv2.imwrite("%s/%s/changed_img/%.6i.png"%(save_path, obj, save_count), result)
		# # cv2.imwrite("%s/%s/depth/%.6i.png"%(save_path, obj, save_count), new_depth)

		# save_count += 1

		# print("object:%s, number:%i"%(obj,save_count))