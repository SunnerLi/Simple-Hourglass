import numpy as np
import cv2

def drawRec(img, ann):
    img = np.copy(img)
    gray_ann = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY)
    num_segment, label_map, component_info_list, centroids = cv2.connectedComponentsWithStats(gray_ann, 4, cv2.CV_32S)
    for i in range(1, num_segment):
        start_point = (component_info_list[i][0], component_info_list[i][1])
        end_point = (component_info_list[i][0] + component_info_list[i][2], component_info_list[i][1] + component_info_list[i][3])
        color = (
            int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][0]), 
            int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][1]), 
            int(revert_ann[int(centroids[i][1])][int(centroids[i][0])][2])
        )
        cv2.rectangle(img, start_point, end_point, color, 2)
    return img

if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    revert_ann = cv2.imread('1.png')
    cv2.imshow('res', drawRec(img, revert_ann))
    cv2.waitKey(0)