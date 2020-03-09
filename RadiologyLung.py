import sys
import glob
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

raw_image_path = 'Data/samples/'
output_directory = 'Data\\outputtojpg'
candidate_file = 'Data\\candidates.csv'
X_log_train = []
X_log_test = []
X_log_val = []

class MHDConverter(object):
    def __init__(self, filename=None, coords=None, path=None):
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.img = None
        self.path = path

    def reset_coords(self, coords):
        self.coords = coords

    def read_mhd_img(self):
        #path = glob.glob(self.path + self.filename + '.mdh')
        path = [raw_image_path+name for name in os.listdir(raw_image_path) if name.endswith(".mhd")]
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_voxel_coords(self):
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution for j in range(len(self.coords))]
        return tuple(voxel_coords)

    def get_subimg(self, width):
        self.read_mhd_img()
        x, y, z = self.get_voxel_coords()
        print(self.image)
        subImg = self.image[int(z), int(y-width/2):int(y+width/2), int(x-width/2):int(x+width/2)]
        return subImg

    def normalizePlanes(self, npzarray):
        maxHU = 400
        minHU = -1000
        npzarray = (npzarray - minHU)/ (maxHU - minHU)
        npzarray[npzarray>1] = 1
        npzarray[npzarray<0] = 0
        return npzarray

    def save_img(self, filename, width):
        img = self.get_subimg(width)
        img = self.normalizePlanes(img)
        Image.fromarray(img*255).convert('L').save(filename)



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

def convert_data(idx, outDir, X_data, width=50):
    converter = MHDConverter(np.asarray(X_data.loc[idx])[0], np.asarray(X_data.loc[idx])[1:], raw_image_path)
    outfile = outDir + str(idx) + '.jpg'
    converter.save_img(outfile, width)

def do_test_train_split(filename):
    """
    Does a test train split if not previously done
    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives)*5, replace=False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')

def buildModel(inputShape):
    model = ks.Sequential()
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
    Parallel(n_jobs=3)(delayed(convert_data)(idx, outDir, X_data) for idx in X_data.index)
    for f in os.listdir(outDir):
        img = f.open()
        cv.imshow("filename", img)
        #do_train_test_split(f)

    model, batchSize, epochs = buildModel('''Insert Image Size Here''')
    model.summary()
    model.fit(x_log_train, x_log_test, batch_size=batchSize, epochs=epochs, validation_split=0.25)


if __name__ == "__main__":
    main()