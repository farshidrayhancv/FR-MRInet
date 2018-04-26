import os
import numpy as np


# assumed "path" ---> images ---> 1.jpg
#                |                2.jpg
#                |
#                ---> annotation --> 1.txt
#                                --> 2.txt

# image count
import cv2


class File_loader:
    path = None

    def __init__(self, path):
        File_loader.path = path

    def check_count(self):
        # to verify that equal number of image and annotation is present
        _, _, files = next(os.walk(File_loader.path + 'input_images'))
        number_of_input = len(files)
        # print(number_of_input)
        _, _, files = next(os.walk(File_loader.path + 'output_images'))
        number_of_output = len(files)
        # print(number_of_output)

        _, _, files = next(os.walk(File_loader.path + 'output_images'))
        number_of_output = len(files)
        # print(number_of_output)
        if number_of_input > 0 and number_of_output > 0:
            if number_of_input == number_of_output:
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
        output_image_path_list = []


        if self.check_count():
            pass
            for root, dirs, files in os.walk(File_loader.path + 'input_images', topdown=False):
                for name in files:
                    # print(os.path.join(root, name))
                    input_image_path_list.append(os.path.join(root, name))

            for root, dirs, files in os.walk(File_loader.path + 'output_images', topdown=False):
                for name in files:
                    # print(os.path.join(root, name))
                    output_image_path_list.append(os.path.join(root, name))

        return input_image_path_list, output_image_path_list

    def get_location(self,annotation_paths):

        x_list = []
        y_list = []
        h_list = []
        w_list = []

        for test_path in annotation_paths:
            # i += 1
            with open(test_path, 'r') as fp:
                fp.readline()  # 1st line is not necessary

                strg = fp.readline().split()

                x = strg[1]
                y = strg[2]
                h = strg[3]
                w = strg[4]

                x_list.append(x)
                y_list.append(y)
                h_list.append(h)
                w_list.append(w)

        return x_list, y_list, h_list, w_list

    def get_labels(self,annotation_paths):
        label_list = []

        for test_path in annotation_paths:
            # i += 1
            with open(test_path, 'r') as fp:
                fp.readline()  # 1st line is not necessary

                strg = fp.readline().split()
                label = strg[0]

                label_list.append(label)

        return label_list

    def load_images(self,input_path,output_path):

        input_images = []

        for i in input_path:
            input_images.append(cv2.imread(i))

        output_images = []

        for i in output_path:
            output_images.append(cv2.imread(i))

        input_images = np.asarray(input_images)
        output_images = np.asarray(output_images)

        if input_images[0].shape == output_images[0].shape:
            return np.asarray(input_images) , np.asarray(output_images)
        else:
            print("Fatal Error !! input and output image size do not match !!!!")

        # path = '/home/farshid/PycharmProjects/RPNplus/Train/'
        # files =File_loader(path=path)
        # image,annotation = files.load_files()




        # for a,b in zip(image,annotation):
        #     print(a,b)
