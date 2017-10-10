import numpy as np
import cv2

img = cv2.imread('1.jpg')
revert_ann = cv2.imread('1.png')
gray_ann = cv2.cvtColor(revert_ann, cv2.COLOR_BGR2GRAY)
# gray_ann = cv2.resize(gray_ann, (int(np.shape(gray_ann)[0]/100), int(np.shape(gray_ann)[1]/100)))

# cv2.imshow('res', gray_ann)
# cv2.waitKey(0)

# img = np.ones([5, 5])
# ann = np.asarray([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 2, 2, 2, 0]], dtype=np.uint8)
# revert_ann = np.asarray([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
# [[0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
# [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0]],
# [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
# [[0, 0, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 0, 0]]])

print(revert_ann)
num_segment, label_map, component_info_list, centroids = cv2.connectedComponentsWithStats(gray_ann, 4, cv2.CV_32S)
print(num_segment)
print(label_map, np.max(label_map))
print(component_info_list)
print(centroids)


for i in range(1, num_segment):
    start_point = (component_info_list[i][0], component_info_list[i][1])
    end_point = (component_info_list[i][0] + component_info_list[i][2], component_info_list[i][1] + component_info_list[i][3])
    color = (
        int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][0]), 
        int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][1]), 
        int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][2])
    )
    print('color: ', color)
    cv2.rectangle(img, start_point, end_point, color, 2)
cv2.imshow('res', img)
cv2.waitKey(0)