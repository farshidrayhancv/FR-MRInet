import numpy

path = 'C:\\Users\\Farshid\\Desktop\\dataset\\roislist\\1.csv'
file = open(path,'r')

roi_list =  file.readline()
roi_list = roi_list.split(",")
roi_list = numpy.asarray(roi_list)

print(type(roi_list))
print(roi_list.shape)

