from __future__ import division, print_function, absolute_import
from ROI.files_loader import File_Loader
import cv2
import os

#
path = os.getcwd() + '/'
files = File_Loader(path=path)

input_image_paths, rois_path_list, annotations_path_list, output_images_path_list = files.load_paths()

# print(input_image_paths.sort())
# print(annotations_path_list)
input_image = files.load_images(input_path=input_image_paths)
rois_files = files.load_rois(input_path=rois_path_list)
output_images = files.load_rois(input_path=output_images_path_list)

# print(x_list[0])
# print(y_list[0])
# print(w_list[0])
# print(h_list[0])
x_list = [60]
y_list = [45]
w_list = [20]
h_list = [20]
#
# for i in range(0,len(output_images)):
#     print(output_images_path_list[i])
print(input_image_paths[0])
img = cv2.imread(input_image_paths[0])
# img = input_image[i]

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     # cv2.putText(img,'face', (x - w, y - h), font, 0.5, (0, 255, 255), 2)
cv2.rectangle(img, (x_list[0], y_list[0]), (x_list[0]+w_list[0], y_list[0]+ h_list[0]), (0, 255, 0), 2)  #
cv2.imshow('Image Detector', img)
cv2.imwrite(img=img,filename='0.png')
cv2.waitKey(0)
# cv2.destroyAllWindows()
