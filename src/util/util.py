from scipy.misc import imread
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import basename, splitext, exists, split


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def safe_mkdir(dir):
    try:
        os.mkdir(dir)
    except OSError as e:
        if e[0] == 17: #Dir already exists
            return
        else:
            raise e

class Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def plot_query(im1, im2, query = "Empty"):
    plt.ion()
    # plt.figure()
    getch = Getch()
    # Plot 2 images
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.subplot(1,2,2)
    plt.imshow(im2)
    # plt.subplot(2,2,3)
    # plt.imshow(seq[:,:,:,index] - seq[:,:,:,index-1])
    plt.show()
    plt.pause(0.002)
    # reject or accept change

    while True:
        print(query + " [y/n]")
        char = getch()
        print(char)
        if char == 'y':
            return True
        elif char == 'n':
            return False

def rmse_error(sparse, gt, validEntries = None):
    """
        Given two images, gives the error between the non-sparse parts
    """
    # import pdb; pdb.set_trace()
    if validEntries is None:
        validEntries = sparse != 0
    validEntries = np.logical_and(validEntries, (gt!=0))

    depthSparse = np.power(sparse, -1)
    E = depthSparse - gt
    E[np.logical_not(validEntries)] = 0
    SE = np.power(E, 2)
    RMSE = np.sqrt(np.sum(SE)/np.sum(validEntries))
    return RMSE, E

def scale_im_invert(im, scale):
    validEntries = im != 0
    ret = np.power(im*scale, -1)
    ret[np.logical_not(validEntries)] = 0
    return ret

def strip_id_from_fn(fn):
    """
        Gets the image id from the filename (as string)
    """
    return splitext(basename(fn))[0]

class GT_getter(object):
    """
        This manages loaded .mat files to minimize load operations
        mode = 0: .mat files, stijn GT
        mode = 1: depth .jpg files
    """

    def __init__(self, path, mode = 0):
        self.path = path
        self.loadedMat = -1
        self.matFile = None
        self.mode = mode

    def fetch_gt_im(self,im_id):
        if self.mode == 0:
            return self.fetch_gt_im_mat(im_id)
        elif self.mode == 1:
            return self.fetch_gt_im_im(im_id)

        return None

    def fetch_gt_im_im(self, im_id):
        gt = imread("%s/%s.png" % (self.path, im_id), flatten = True)
        gt = gt.astype(np.float32)/255 * 8
        return gt

    def fetch_gt_im_mat(self,im_id):
        """
            Finds the ground truth depth information corresponding to the image id
        """
        im_id = int(im_id) + 1 # In original dataset id starts with 1
        # The dataset is stored in separate .mat files like this
        # esat_id_(id+1999).mat, ex: esat_2001_4000
        offset = 2e3
        match = None
        for setNo in xrange(11):
            startID = setNo*offset + 1
            endID = (setNo + 1)*offset
            if startID <= im_id <= endID:
                # Match found!
                match = self.path + "/esat_%d_%d.mat" % (startID, endID)
                break

        if match is None:
            return None

        if self.loadedMat != match:
            self.matFile = loadmat(match)
            self.loadedMat = match

        matId = (im_id-1) % offset
        # Load mat file
        return self.matFile["data"]["depth_map"][:,:,matId]

class RMSE_constructor(object):
    def __init__(self, sparse, gt):
        self.sparse = sparse
        self.gt = gt
        # self.validEntries = sparse != 0
        self.validEntries = np.logical_and(sparse != 0, sparse != 1)
        # plt.imshow(self.validEntries); plt.show()

    def error(self, scale):
        (err, _) = rmse_error(self.sparse/scale, self.gt, validEntries = self.validEntries)
        return err

    def error_grad(self, scale):
        # FIXME 
        N = np.sum(self.validEntries)
        topM = np.multiply((self.sparse*scale - self.gt), self.sparse)
        topM[np.logical_not(self.validEntries)] = 0
        top = 1.0/(N) * np.sum(topM)
        (bot, _) = rmse_error(self.sparse*scale, self.gt, self.validEntries)

        print("Grad: %1.6f / %1.6f = %1.6f" % (top,bot,(top/bot)))
        return np.asarray([top/bot])


def normalized_cm_viz(im1, im2, names = None):
    if names is None:
        names = ("Im1", "Im2")

    # get Max im1
    max1 = np.max(im1)
    # get max im2
    max2 = np.max(im2)
    # choose global max
    gMax = max(max1,max2)
    # Normalize both images using max
    # im1 = im1/gMax
    # im2 = im2/gMax
    # Viz
    plt.subplot(131); plt.imshow(im1, vmin = 0, vmax = gMax); plt.title(names[0])
    plt.subplot(132); plt.imshow(im2, vmin = 0, vmax = gMax); plt.title(names[1])
    plt.subplot(133);
    plt.colorbar()
    plt.show()