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

raw_image_path = '../../data/samples/'
output_directory = '../../data/outputtojpg'
candidate_file = '../data/candidates.csv'

class MHDConverter(object):
    def __init__(self, filename= None, coords= None, path= None):
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.img = None
        self.path = None

    def reset_coords(self, coords):
        self.coords = coords

    def read_mhd_img(self):
        path = glob.glob(self.path + self.filename + '.mdh')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def get_voxel_coords(self):
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        voxel_coords = [np.absolute(self.coords[j]-origin[j]/resolution for j in range(len(self.coords)))]
        return tuple(voxel_coords)

    def get_subimg(self, width):
        self.read_mhd_img()
        x, y, z = self.get_voxel_coords()
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

def main():

if __name__ == "__main__":
    main()