import numpy as np
import os
import argparse

import cv2

print(cv2.__version__)

mouse_x = 0
mouse_y = 0
class_index = 0
img_index = 0
circles = np.zeros((50,2))
counter = 0
img_dir = "images/"
seg_dir = "seg_txt/"
is_same = False
image_list = []
class_point = (1000,1000)

WITH_QT = True
try:
    cv2.namedWindow("Test")
    cv2.displayOverlay("Test", "Test QT", 1000)
except:
    WITH_QT = False
# cv2.destroyAllWindows()

# bbox_thickness = 2

# parser = argparse.ArgumentParser(description='YOLO v2 Bounding Box Tool')
# parser.add_argument('--format', default='yolo', type=str, choices=['yolo', 'voc'], help="Bounding box format")
# parser.add_argument('--sort', action='store_true', help="If true, shows images in order.")
# parser.add_argument('--cross-thickness', default='1', type=int, help="Cross thickness")
# parser.add_argument('--bbox-thickness', default=bbox_thickness, type=int, help="Bounding box thickness")

# args = parser.parse_args()

# make normalization for all points and returns class_index and list of this points
def yolo_format_seg(class_index, circles, width, height):
    right_circles = []
    for a in circles:
        if(a[0] != 0):        
            x = a[0] / float(width)
            y = a[1] / float(height)
            right_circles.append((x,y))
    circles = np.zeros((50,2))
    return class_index , right_circles

def change_img_index(x):
    global img_index, img
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path)
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Showing image "
                                    "" + str(img_index) + "/"
                                    "" + str(last_img_index), 1000)
    else:
        print("Showing image "
                "" + str(img_index) + "/"
                "" + str(last_img_index) + " path:" + img_path)

def change_class_index(x):
    global class_index
    class_index = x
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Selected class "
                                "" + str(class_index) + "/"
                                "" + str(last_class_index) + ""
                                "\n " + class_list[class_index],3000)
    else:
        print("Selected class :" + class_list[class_index])

def change_state_augmentation(x):
    # print(key_state)
    global key_state
    if x:
        key_state = True
    else:
        key_state = False


def get_txt_path(img_path):
    img_name = os.path.basename(os.path.normpath(img_path))
    img_type = img_path.split('.')[-1]
    return seg_dir + img_name.replace(img_type, 'txt')

def save_bb(txt_path, class_idx , points):
    myfile = open(txt_path, 'a')
    myfile.write(str(class_idx) + " " )
    for a in points:
        myfile.write(str(a[0]) + " " + str(a[1]) + " ") # append line
    myfile.write("\n")

def draw_text_from_file(tmp_img , cls_ind , all_pointss):
    font = cv2.FONT_HERSHEY_SIMPLEX
    class_point1 = (1000,1000)
    for a in all_pointss:
        if(a[1] < class_point1[1]):
            class_point1 = a
    cv2.putText(tmp_img, class_list[int(cls_ind)], class_point1 , font, 0.6, class_rgb[int(cls_ind)].tolist(), 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return tmp_img
def draw_text(tmp_img, text, color):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, class_point , font, 0.6, color, 1, cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return tmp_img

def yolo_to_x_y_seg(x_center, y_center, width, height):
    x_center = float(x_center)
    y_center = float(y_center)
    x_center *= width
    y_center *= height
    return (int(x_center) ,int(y_center))

def draw_seg_from_file(tmp_img, txt_path, text , color , width, height):
    valuesList = []
    file_points = []
    all_points = []
    counter = 0
    global from_file
    from_file = True
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            content = f.readlines()
        for line in content:
            valuesList = line.split(" ")
            if True: # args.format == 'yolo':
                class_index1 = valuesList[0]
                file_points = valuesList[1:-1]
                for a in range(len(file_points)):
                    #print("1")
                    if counter % 2 == 1: 
                        counter += 1
                        continue
                    all_points.append(yolo_to_x_y_seg(file_points[a], file_points[a+1] , width, height))
                    #print("2")
                    counter += 1
                for a in range(len(all_points)):
                    if a == len(all_points) - 1:
                        cv2.line(tmp_img , all_points[a] , all_points[0],class_rgb[int(class_index1)].tolist(),2)
                        continue
                    cv2.line(tmp_img , all_points[a] , all_points[a+1],class_rgb[int(class_index1)].tolist(),2) 
                tmp_img = draw_text_from_file(tmp_img,class_index1 , all_points)
                all_points = []
    return tmp_img        

def mousePoints(event,x,y,flags,params):
    global counter ,circles , mouse_x , mouse_y , is_same, class_point
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(1)
        circles[counter] = x,y
        cv2.circle(img,(x,y),2,color,cv2.FILLED)
        if(class_point[1] > y):
            class_point = (x,y)
        file = open("segment.txt" , "a")
        print(2)
        file.write(f'({str(x)} , {str(y)})')
        print("counter = " + str(counter))
        if counter > 0:
            print(3)
            print(circles)
            pt1 = (int(circles[counter-1][0]), int(circles[counter-1][1]))
            pt2 = (int(circles[counter][0]), int(circles[counter][1]))
            cv2.line(img, pt1, pt2, color, 2)
            # cv2.line(img , circles[counter-1] , circles[counter],color,2)  
        if counter != 0:
            print(4)
            if((circles[0][0] - 5 < circles[counter][0] < circles[0][0] + 5) and (circles[0][1] - 5 < circles[counter][1] < circles[0][1] + 5)):
                print(5)
                file.write("\n")
                is_same = True
                counter = -1
        counter += 1

# create empty .txt file for each of the images if it doesn't exist already
for f in os.listdir(img_dir):
    f_path = os.path.join(img_dir, f)
    test_img = cv2.imread(f_path)
    if test_img is not None:
        image_list.append(f_path)

last_img_index = len(image_list) - 1

if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)

for img_path in image_list:
    txt_path = get_txt_path(img_path)
    if not os.path.isfile(txt_path):
        open(txt_path, 'a').close()

# load class list
with open('c:\\Users\\AMIRA\\Desktop\\graduate_proj\\graduate_proj\\Face_recognize\\class_list.txt') as f:
    class_list = f.read().splitlines()

last_class_index = len(class_list) - 1

# Make the class colors the same each session
# The colors are in BGR order because we're using OpenCV
class_rgb = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
class_rgb = np.array(class_rgb)
# If there are still more classes, add new colors randomly
num_colors_missing = len(class_list) - len(class_rgb)
if num_colors_missing > 0:
    more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
    class_rgb = np.vstack([class_rgb, more_colors])

# create window
WINDOW_NAME = 'Bounding Box Labeler'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME,500, 500)
cv2.setMouseCallback(WINDOW_NAME, mousePoints)

i = 0

# # selected image
TRACKBAR_IMG = 'Image'
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, i, last_img_index, change_img_index)

# selected class
TRACKBAR_CLASS = 'Class'
if last_class_index != 0:
    cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, i, last_class_index, change_class_index)

key_state = False

# TRACKBAR_AUG = f'Augmentation : {key_state}'
# cv2.createTrackbar(TRACKBAR_AUG, WINDOW_NAME, i, 1, change_state_augmentation)

change_img_index(i)

if WITH_QT:
    cv2.displayOverlay(WINDOW_NAME, "Welcome!\n Press [h] for help.", 4000)
print(" Welcome!\n Select the window and press [h] for help.")

# Function to create the trackbar based on the key state
# def create_trackbar(key_state):
#     trackbar_name = 'State: ON' if key_state else 'State: OFF'
#     cv2.createTrackbar(trackbar_name, WINDOW_NAME, 0, last_img_index, change_img_index)
# # Initial trackbar
# create_trackbar(key_state)


while True:
    tmp_img = img.copy()
    # state_text = f"State: {key_state}"  # Create the text to display the True/False state
    # cv2.putText(tmp_img, state_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # TRACKBAR_AUG = f'Augmentation : {key_state}'

    height, width = tmp_img.shape[:2]
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)
    text = class_list[class_index]
    color = class_rgb[class_index].tolist()
    tmp_img = draw_seg_from_file(tmp_img, txt_path, text , color , width, height)
    if is_same:
        if True: #args.format == 'yolo':
            cls_idx , points = yolo_format_seg(class_index, circles, width, height)
        save_bb(txt_path, cls_idx , points)
        img = draw_text(tmp_img, text, color)
        class_point = (width,height)
        is_same = False
        circles = np.zeros((50,2))
    else:
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "Selected label: " + class_list[class_index] + ""
                                "\nPress [w] or [s] to change.", 120)
    cv2.imshow(WINDOW_NAME, tmp_img)
    cv2.waitKey(1)
    key = cv2.waitKey(51) & 0xFF

    # print(key_state)
    # Press 't' to toggle between True and False
    if key == ord('t'):
        key_state = not key_state

        cv2.destroyWindow(WINDOW_NAME)  # Remove the window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)  # Recreate the window
        cv2.resizeWindow(WINDOW_NAME, 500, 500)
        # create_trackbar(key_state)  # Create the trackbar with the new name

        # cv2.setTrackbarPos('State: ON' if not key_state else 'State: OFF', WINDOW_NAME, 0)
        # create_trackbar(key_state)

    # pressed_key = cv2.waitKey(50)
    if key == ord('q'):
        break
