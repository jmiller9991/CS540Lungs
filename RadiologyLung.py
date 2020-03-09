import sys
import os
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
import cv2 as cv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow.keras.utils
import random
import math as m

raw_image_path = 'Data/samples/'
output_directory = 'Data/outputtojpg/'
candidate_file = 'Data/candidates.csv/'
X_log_train = []
X_log_test = []


class CTScan(object):
    """
	A class that allows you to read .mhd header data, crop images and
	generate and save cropped images

    Args:
    filename: .mhd filename
    coords: a numpy array
	"""

    def __init__(self, filename=None, coords=None, path=None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self.path = path

    def reset_coords(self, coords):
        """
        updates to new coordinates
        """
        self.coords = coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        # path = glob.glob(self.path + self.filename + '.mhd')
        path = [raw_image_path + name for name in os.listdir(raw_image_path) if name.endswith(".mhd")]
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        voxel_coords = [np.absolute(self.coords[j] - origin[j]) / resolution[j] \
                        for j in range(len(self.coords))]
        return tuple(voxel_coords)

    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image

    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensiona
        """
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y - width / 2):int(y + width / 2), \
                   int(x - width / 2):int(x + width / 2)]
        return subImage

    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
        return npzarray

    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)
        Image.fromarray(image * 255).convert('L').save(filename)


def log(image):
    gauss = cv.GaussianBlur(image, (3, 3), 0)
    laplacian = cv.Laplacian(gauss, cv.CV_32F, 3, 0.25)
    return laplacian

def hog(image, size, blockSize, blockStride, cellSize, nbins, winStride, padding, locations):
    histOrGrad = cv.HOGDescriptor(size, blockSize, blockStride, cellSize, nbins)

    hist = histOrGrad.compute(image, winStride, padding, locations)
'''
def computeOneFeatures(image, feature, startIndex):
    magImg = np.zeros((256, 256, 1), dtype="float64")
    angleImg = np.zeros((256, 256, 1), dtype="float64")
    gauss = cv.GaussianBlur(image, (3, 3), 0)
    sobelX = cv.Sobel(gauss, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(gauss, cv.CV_64F, 0, 1)
    sumgrad = 0.0

    cv.cartToPolar(sobelX, sobelY, magImg, angleInDegrees=True)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            index = np.floor((angleImg[x, y]) / 10)
            feature[0, startIndex + index] += magImg[x, y]
            sumgrad += magImg[x, y]

    if sumgrad > 0:
        for i in range(36):
            feature[0, startIndex + i] /= sumgrad

def getFeatureLength(regionSideCount):
    return (regionSideCount * regionSideCount) * 36

def computeFeatures(image, regionSideCount, feature):
    regionSize = image[1]/regionSideCount
    index = 0

    for i in range (feature[1]):
        feature[0, i] = 0

    for i in i < regionSideCount:
        for j in j < regionSideCount:
            subImg = image(cv.rectangle(j * regionSize), (i * regionSize), regionSize, regionSize)
            computeOneFeatures(subImg, feature, index)
            index += 36
'''

def create_data(idx, outDir, X_data, width=50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    scan = CTScan(np.asarray(X_data.loc[idx])[0], \
                  np.asarray(X_data.loc[idx])[1:], raw_image_path)
    outfile = outDir + str(idx) + '.jpg'
    scan.save_image(outfile, width)


def do_test_train_split(filename):
    """
    Does a test train split if not previously done

    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class'] == 1].index
    negatives = candidates[candidates['class'] == 0].index

    ## Under Sample Negative Indexes
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives) * 5, replace=False)

    candidatesDf = candidates.iloc[list(positives) + list(negIndexes)]

    X = candidatesDf.iloc[:, :-1]
    y = candidatesDf.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
                                                      test_size=0.25, random_state=42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')


def buildModel(inputShape):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.2))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    batchSize = 32
    epochs = 30

    return model, batchSize, epochs


def splitLogMethod(dir):
    X_data_orig = []
    X_train = []
    X_test = []
    path = [open(dir + name) for name in os.listdir(dir)]
    X_data_orig.append(log(path))
    cv.imshow("Logs", X_data_orig[0])

    random.seed(420)
    for i in range(m.floor(X_data_orig.__sizeof__()*0.75)):
        temp1 = random.choice(X_data_orig)
        X_data_orig.remove(temp1)
        X_train.append(temp1)

    for i in range(X_data_orig.__sizeof__()):
        temp2 = random.choice(X_data_orig)
        X_data_orig.remove(temp2)
        X_test.append(temp2)

    return X_train, X_test


def convert_pickles_outdir(data, idx):
    inp = cv.imread(data +'/image_' + str(idx) + '.jpg')
    Image.fromarray(inp).convert('L').save(output_directory +\
                                           'image_' + str(idx) + '.jpg')


def main():
    if len(sys.argv) < 2:
        raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
        mode = sys.argv[1]
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Argument not recognized. Has to be train, test or val')

    inpfile = mode + 'data'
    outDir = mode + '/image_'

    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split(candidate_file)
    X_data = pd.read_pickle(inpfile)
    Parallel(n_jobs=3)(delayed(create_data)(idx, outDir, X_data) for idx in X_data.index)

    X_train = pd.read_pickle('traindata')
    y_train = pd.read_pickle('trainlabels')
    augIndexes = X_train[y_train == 1].index
    Parallel(n_jobs=3)(delayed(convert_pickles_outdir)('train', idx) for idx in augIndexes)

    X_test = pd.read_pickle('testdata')
    y_test = pd.read_pickle('testlabels')
    augIndexes = X_train[y_train == 1].index
    Parallel(n_jobs=3)(delayed(convert_pickles_outdir)('test', idx) for idx in augIndexes)

    X_val = pd.read_pickle('valdata')
    y_val = pd.read_pickle('vallabels')
    augIndexes = X_train[y_train == 1].index
    Parallel(n_jobs=3)(delayed(convert_pickles_outdir)('val', idx) for idx in augIndexes)

    x_log_train, x_log_test = splitLogMethod(output_directory)

    model, batchSize, epochs = buildModel((50, 50))
    model.summary()
    model.fit(x_log_train, x_log_test, batch_size=batchSize, epochs=epochs, validation_split=0.25)


if __name__ == "__main__":
    main()