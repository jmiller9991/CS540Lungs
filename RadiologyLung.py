import sys
import glob
import os
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk

raw_image_path = '../ ../ data/samples/*/'
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

    def save_image(self, filename, width):
        img = self.get_subimg(width)
        img = self.normalizePlanes(img)
        Image.fromarray(img*255).convert('L').save(filename)

