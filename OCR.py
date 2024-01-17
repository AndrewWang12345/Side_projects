import cv2
import numpy
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

DATA_DIRECTORY = 'C:/Users/andrew/Downloads/training_data/'
TEST_DATA_FILENAME = DATA_DIRECTORY + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIRECTORY + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIRECTORY + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIRECTORY + 'train-labels.idx1-ubyte'

def read_images(filename):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_columns = int.from_bytes(f.read(4), 'big')
        for image_index in range(10000):

            image = []
            for row_index in range(n_rows*n_columns):
                pixel = int.from_bytes(f.read(1), "big")
                #print(pixel)
                if pixel != 0:
                    pixel = 255
                image.append(pixel)
            images.append(image)
        return images

def read_labels(filename):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_labels = int.from_bytes(f.read(4), 'big')
        for label_index in range(10000):
            #print("a")
            label = int.from_bytes(f.read(1), "big")
            labels.append(label)
        return labels

def distance(x, y):
    #print(sum([(int.from_bytes(x_i, 'big') - int.from_bytes(y_i, 'big')) ** 2 for x_i, y_i in zip(x, y)]))
    return sum([(x_i - y_i) ** 2 for x_i, y_i in zip(x, y)]) ** 0.5

def get_training_distances_for_test_sample(X_train, test_sample):
    return[distance(train_sample, test_sample) for train_sample in X_train]

def knn(X_train, Y_train, X_test, k = 5):
    Y_pred = []
    for test_sample_index, test_sample in enumerate(X_test):
        training_distances = get_training_distances_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances), key = lambda x: x[1]
            )
        ]
        candidates = [
            Y_train[index] for index in sorted_distance_indices[:k]
        ]
        print(f'Point is {candidates}')
        Y_sample = 5
        Y_pred.append(Y_sample)
    return Y_pred

def main():
    # Use a breakpoint in the code line below to debug your script.
    image = cv2.imread('C:/Users/andrew/Downloads/training_data/sudoku.png')
    #image = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = numpy.zeros((gray.shape),numpy.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    div = numpy.float32(gray) / (close)
    res = numpy.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

    #finding the sudoku square
    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    contour,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    x,y,w,h = cv2.boundingRect(best_cnt)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,3)
    res = cv2.bitwise_and(res,mask)
    print(y)
    print(y+h)
    print(x)
    print(x+w)
    res = res[y:y+h, x:x+w]
    res = cv2.resize(res, (800,800))


    #finding horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
    dx = cv2.Sobel(res, cv2.CV_16S, 0,1)
    dx = cv2.convertScaleAbs(dx)
    ret, dx = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dx = cv2.morphologyEx(dx, cv2.MORPH_DILATE, vertical_kernel)
    contours, hierarchy = cv2.findContours(dx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w/h > 6:
            cv2.drawContours(dx, [contour], 0, 255, -1)
        else:
            cv2.drawContours(dx, [contour], 0, 0, -1)
    dx = cv2.morphologyEx(dx, cv2.MORPH_CLOSE, None, iterations=2)


    #finding vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,10))
    dy = cv2.Sobel(res, cv2.CV_16S,1,0)
    dy = cv2.convertScaleAbs(dy)
    ret,dy = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dy = cv2.morphologyEx(dy,cv2.MORPH_DILATE,vertical_kernel)
    contours, hierarchy = cv2.findContours(dy,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if h/w > 7:
            cv2.drawContours(dy,[contour],0,255,-1)
        else:
            cv2.drawContours(dy, [contour], 0,0,-1)
    dy = cv2.morphologyEx(dy,cv2.MORPH_CLOSE, None, iterations = 2)


    #combining the two pictures
    final_image = cv2.bitwise_and(dx,dy)
    contours, hierarchy = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h / w > 3 or w/h > 3:
            cv2.drawContours(final_image, [contour], 0, 0, -1)
        else:
            cv2.drawContours(final_image, [contour], 0, 255, -1)


    #finding centroids
    contours, hierarchy = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gridPoints = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            point = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        else:
            point = [0,0]
        gridPoints.append(point)

        #cv2.circle(image, (point[0], point[1]), 1, (255, 255, 255), -1)
    gridPoints = sorted(gridPoints, key=lambda x: x[1])
    print(len(gridPoints))
    gridPoints2D = numpy.zeros((10, 10, 2))
    for i in range(10):
        for j in range(10):
            gridPoints2D[i][j][0] = int(gridPoints[10 * i + j][0])
            gridPoints2D[i][j][1] = int(gridPoints[10 * i + j][1])
        temp = sorted(gridPoints2D[i], key=lambda x:x[0])
        a = 0
        tempcopy = numpy.copy(temp)
        for b in tempcopy:
            #print(b)
            gridPoints2D[i][a][0] = b[0]
            gridPoints2D[i][a][1] = b[1]
            a += 1
    #print(gridPoints)


    #separating the images
    image_array = []
    for i in range(9):
        for j in range(9):
            pts1 = numpy.float32([[gridPoints2D[i][j][0], gridPoints2D[i][j][1]], [gridPoints2D[i][j+1][0], gridPoints2D[i][j+1][1]], [gridPoints2D[i+1][j][0], gridPoints2D[i +1][j][1]], [gridPoints2D[i+1][j+1][0], gridPoints2D[i+1][j+1][1]]])
            pts2 = numpy.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(res,M,(300,300))
            #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            thresh, dst = cv2.threshold(dst, 80, 255, cv2.THRESH_BINARY)
            #cv2.drawContours(dst, [contour], 0, 0, -1)
            image_array.append(dst)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    for i in range(len(image_array)):
        image_array[i] = cv2.morphologyEx(image_array[i], cv2.MORPH_CLOSE, kernel)
        image_array[i] = image_array[i][20:280, 20:280]
        image_array[i] = cv2.resize(image_array[i], (28,28))
        # cv2.imshow("",image_array[i])
        # cv2.waitKey(0)
    X_test = []
    for i in image_array:
        X_test.append(numpy.array(i).ravel())

    #print(len(X_test))
    numpy.zeros((1, 784), dtype=bytes)
    for i in range(len(X_test)):
        for j in range(len(X_test[i])):
            if X_test[i][j] == 255:
                num = 0
                X_test[i][j] = num
            else:
                num = 255
                X_test[i][j] = num
    X_train = read_images(TRAIN_DATA_FILENAME)#training images
    Y_train = read_labels(TRAIN_LABELS_FILENAME)#answers to training images
    #print(X_train[5])
    #print(Y_train[5])
    zero = 0
    # Y_train.append(zero.to_bytes(1,"big"))
    # Y_train.append(zero.to_bytes(1,"big"))
    # Y_train.append(zero.to_bytes(1,"big"))
    # Y_train.append(zero.to_bytes(1,"big"))
    # Y_train.append(zero.to_bytes(1,"big"))
    #X_train = extract_features(X_train)
    # X_train.append(numpy.zeros((1, 784), dtype=bytes))
    # X_train.append(numpy.zeros((1, 784), dtype=bytes))
    # X_train.append(numpy.zeros((1, 784), dtype=bytes))
    # X_train.append(numpy.zeros((1, 784), dtype=bytes))
    # X_train.append(numpy.zeros((1, 784), dtype=bytes))
    #X_test = extract_features(X_test)

    knn(X_train, Y_train, X_test, 5)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
