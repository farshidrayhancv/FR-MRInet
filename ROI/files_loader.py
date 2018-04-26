import os
import numpy as np
# from PIL import Image
# assumed "path" ---> images ---> 1.jpg
#                |                2.jpg
#                |
#                ---> annotation --> 1.txt
#                                --> 2.txt

# image count
import cv2


class File_Loader:
    path = None
    number_of_files = 0

    def __init__(self, path):
        File_Loader.path = path

    def check_count(self):
        # to verify that equal number of image and annotation is present
        _, _, files = next(os.walk(File_Loader.path + 'input_images\\'))
        number_of_input = len(files)
        self.number_of_files = len(files)
        # print(number_of_input)
        _, _, files = next(os.walk(File_Loader.path + 'annotations\\'))
        number_of_annotations = len(files)

        _, _, files = next(os.walk(File_Loader.path + 'output_images\\'))
        number_of_output_images = len(files)
        #
        # _, _, files = next(os.walk(File_Loader.path + 'annotations'))
        # number_of_annotations = len(files)
        # print(number_of_output)

        if number_of_input > 0 and number_of_annotations > 0 and number_of_output_images > 0:
            if number_of_input == number_of_annotations or number_of_input == number_of_output_images:
                return True
            else:
                print('number of input and output do not match')
        else:
            print("no files found")


            # return False

            # for root, dirs, files in os.walk(path, topdown=False):
            #     for name in files:
            #         print(os.path.join(root, name))
            #     for name in dirs:
            #         print(os.path.join(root, name))

    def load_paths(self):

        input_image_path_list = []
        annotations_path_list = []
        rois_path_list = []
        output_images_list = []
        rois_array_list = []

        self.check_count()

        for i in range(1, self.number_of_files + 1):
            # i = 1
            strg = self.path + 'input_images\\' + str(i) + '.png'
            input_image_path_list.append(strg)

            strg = self.path + 'rois\\' + str(i) + '.png'
            rois_path_list.append(strg)
            # print(strg)

            strg = self.path + 'annotations\\' + str(i) + '.csv'
            annotations_path_list.append(strg)

            strg = self.path + 'output_images\\' + str(i) + '.png'
            # print(strg)
            output_images_list.append(strg)

            strg = self.path + 'roislist\\' + str(i) + '.csv'
            # print(strg)
            rois_array_list.append(strg)

            # print(strg)
        #

        return input_image_path_list, rois_path_list, annotations_path_list, output_images_list, rois_array_list

    def get_location(self, annotation_paths):

        x_list = []
        y_list = []
        w_list = []
        h_list = []

        for test_path in annotation_paths:
            # i += 1
            with open(test_path, 'r') as fp:
                # fp.readline()  # 1st line is not necessary

                strg = fp.readline().strip().split(',')

                x = strg[0]
                y = strg[1]
                w = strg[2]
                h = strg[3]

                x_list.append(x)
                y_list.append(y)
                w_list.append(h)
                h_list.append(w)
        x_list = list(map(int, x_list))
        y_list = list(map(int, y_list))
        w_list = list(map(int, w_list))
        h_list = list(map(int, h_list))

        return x_list, y_list, h_list, w_list  # this mistake was done purposefully !!!

    def get_labels(self, annotation_paths):
        label_list = []

        for test_path in annotation_paths:
            # i += 1
            with open(test_path, 'r') as fp:
                fp.readline()  # 1st line is not necessary

                strg = fp.readline().split()
                label = strg[0]

                label_list.append(label)

        return label_list

    def load_images(self,input_path):

        input_images = []

        for i in input_path:
            input_images.append(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY))

        return np.asarray(input_images)

    def load_rois_array(self,input_path):

        rois = []

        for i in input_path:
            file = open(i, 'r')
            roi_list = file.readline()
            roi_list = roi_list.split(",")
            rois.append(roi_list)

        return np.asarray(rois)


    def load_rois(self, input_path):

        rois = []

        for i in input_path:
            rois.append(cv2.imread(i,1))


        return np.asarray(rois)
    def load_output_images(self, input_path):

        output_images = []

        for i in input_path:
            x = cv2.imread(i,1)
            output_images.append(x)
            # cv2.imshow('',x)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        return np.asarray(output_images)

        # output_images = []
        #
        # for i in output_path:
        #     output_images.append(cv2.imread(i))
        #
        # input_images = np.asarray(input_images)
        # output_images = np.asarray(output_images)
        #
        # if input_images[0].shape == output_images[0].shape:
        #     return np.asarray(input_images) , np.asarray(output_images)
        # else:
        #     print("Fatal Error !! input and output image size do not match !!!!")

        # path = '/home/farshid/PycharmProjects/RPNplus/Train/'
        # files =File_loader(path=path)
        # image,annotation = files.load_files()




        # for a,b in zip(image,annotation):
        #     print(a,b)
