from random import randrange, sample
from os.path import join
from skimage import io, color
from matplotlib.pyplot import imshow, plot
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def nearest_square(x):
    return (int(np.floor(x**0.5)), int(np.floor(x**0.5) + 1))
    

def grid_arrange(x): 
    res = []
    for i in range(-(x-1),0): 
        if ((x%-i)==0.0): res.append(-i)
    
    gsize = (x,1)
    if (len(res)>0): 
        res = np.array(res).astype(int)
        div = (x / res).astype(int)
        dif = np.abs(res-div).astype(int)
        m = dif.argmin()
        gsize = (res[m], div[m])

    if (np.abs(gsize[0] - gsize[1]) > x/3):
        gsize = nearest_square(x)
        
    return gsize

class Image(object):
    def __init__(self, img, filename): 
        self.img, self.filename = img, filename

def Load(filename): return io.imread(filename)


def plot_grid(gridy, gridx, func, **kwargs):
    fig, axs = plt.subplots(gridy, gridx, **kwargs)
    axs = np.array(axs).reshape(gridy, gridx)
    for r in range(gridy):
        for c in range(gridx):
            func(r, c, r*gridx + c, fig, axs)
    plt.show()
    
    
def plot_corners(image, corners2D, axs=None):
    pts = np.array(corners2D).reshape(len(corners2D), 2)
    if axs is None: axs = plt.subplot()
    axs.imshow(image)    
    axs.plot(pts[:,0], pts[:,1], 'rx')
    
        
def FindCorners(calib_images, nSqr=(9,6)):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    nx, ny = nSqr
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    inds      = []
    for ind, img in enumerate(calib_images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, nSqr, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            # Refine the pixel points
            corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # update data
            objpoints.append(objp)
            imgpoints.append(corners2)    
            inds.append(ind)
    return objpoints, imgpoints, np.take(calib_images, inds, axis=0)


def calcCalibParams(objpoints, imgpoints, imgs):
    imsize = (imgs[0].shape[:2])[::-1]
    ret = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)
    ret, mtx, dist, rvecs, tvecs = ret
    return mtx, dist


def undistort(img, mtx, dist, retRoi=True):
    imsize = (img.shape[:2])[::-1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist, imsize,1, imsize)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)    
    x,y,w,h = roi
    ret = (dst, dst[y:y+h, x:x+w]) if retRoi else dst
    return ret


def runCameraCalib(calib_images, draw=False):
    # Find Corners
    objpoints, imgpoints, imgs = FindCorners(calib_images)    
    # Calculate Calib Matrix
    matrix, dist = calcCalibParams(objpoints, imgpoints, imgs)
    # Undistort Images
    undist_imgs = np.array([undistort(img, matrix, dist) for img in imgs])
    undist_imgs, undist_roi_imgs = undist_imgs[:,0], undist_imgs[:,1]

    if (draw):
        # Plot some images        
        n_plot = 5
        grid_imgs = [np.hstack((imgs[i], undist_imgs[i])) for i in sample(range(len(imgs)), n_plot)]
        plot_grid(5,1, lambda r, c, i, fig, axs: axs[r,c].imshow(grid_imgs[i]))
    
    return matrix, dist, objpoints, imgpoints, imgs, undist_imgs, undist_roi_imgs    

#%%


def plot_hls_hsv(imHLS, imsHSV, plot_inds=[2,3,6]):
    ims1, ims2 = imHLS, imsHSV
    
    x = ims1[0].shape[1]
    ticks = np.arange(x/8, x, x/4)
    lables = ['Gray', 'Hue', 'Saturation', 'Value']
    for i, inds in enumerate(plot_inds):
    
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20,6))
    
        axs[0].imshow(ims1[inds], cmap='gray')
        axs[0].set_xticks(ticks)
        axs[0].set_xticklabels(['Gray', 'Hue', 'Lightness', 'Saturation'])
        axs[0].set_yticks([])
        
        axs[1].imshow(ims2[inds], cmap='gray')    
        axs[1].set_xticks(ticks)
        axs[1].set_xticklabels(['Gray', 'Hue', 'Saturation', 'Value'])
        axs[1].set_yticks([])
    
#plot_grid(len(plot_inds),1, lambda r,c,i,fig,axs: axs[r][c].imshow(ims[i], cmap='gray'), figsize=(30,40))

#%% Going with Saturation from HLS

def imshow(ax, img, name, cmap=None):
    w = img.shape[1]//2    
    ax.imshow(img, cmap=cmap)
    ax.set_xticks([w])
    ax.set_xticklabels([name])
    ax.set_yticks([])


def imshow_row(imgs, names, cmap=None, figsize=(30,20)):
    n = len(imgs)
    fig, axs = plt.subplots(1,n, figsize=figsize)
    for ax, img, name in zip(axs, imgs, names): imshow(ax, img, name, cmap=cmap)
    plt.show()
    

def findEdges(gray, ksize=7, thresh_rng=(0, 255)):
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize))

    sobelx /= (sobelx.max()/255.0)
    sobely /= (sobely.max()/255.0)  
    
#    imshow_row((gray, sobelx, sobely), ('Original', 'SobelX_'+str(ksize), 'SobelY_'+str(ksize)), cmap='gray')

    thresh = np.array(thresh_rng).astype(np.float64)
    sobelx_thresh = np.zeros_like(sobelx).astype(bool)
    sobely_thresh = np.zeros_like(sobely).astype(bool)
    sobelx_thresh[(sobelx <= thresh[1]) & (sobelx >= thresh[0])] = 1
    sobely_thresh[(sobely <= thresh[1]) & (sobely >= thresh[0])] = 1    
    
#    imshow_row((gray, sobelx_thresh, sobely_thresh), ('Original', 'SobelX_'+str(ksize)+'_Thresh'+str(thresh_rng), 'SobelY_'+str(ksize)+'_Thresh'+str(thresh_rng)), cmap='gray')

    return sobelx.astype(np.uint8), sobely.astype(np.uint8), sobelx_thresh, sobely_thresh



class BirdsEye(object):
    
    def CalcDstCoords(topL, topR, botR, botL):
        d_topL = (botL[0], 0)
        d_topR = (botR[0], 0)
        return d_topL, d_topR, botR, botL

    def GetTransform(botL, topL, topR, botR):
        src = np.array((topL, topR, botR, botL));
        # Get the ideal destination coords
        dst = np.array(BirdsEye.CalcDstCoords(topL, topR, botR, botL));
        # Forward transform matrix
        matrix  = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        # Inverse Transform matrix
        inv_mat = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
        return src, dst, matrix, inv_mat #np.matrix(matrix), np.matrix(inv_mat)

    def TransformImage(img, botL, topL, topR, botR, noplot=True):

        src, dst, matrix, inv_mat = BirdsEye.GetTransform(botL, topL, topR, botR)
        imgBE = cv2.warpPerspective(img, matrix, img.shape[::-1], flags=cv2.INTER_LINEAR)

        if not noplot:     
            npts = len(src)
            # Arrange points to line points pair
            orgP = np.array([[src[i], src[(i+1) % npts]] for i in range(npts)]).reshape(8,2)
            dstP = np.array([[dst[i], dst[(i+1) % npts]] for i in range(npts)]).reshape(8,2)
            # Overlay on src -> dst outline
            plt.imshow(img, cmap='gray')
            plt.plot(orgP[:,0], orgP[:,1], 'r--')
            plt.plot(dstP[:,0], dstP[:,1], 'y--')        
            plt.show()
            
            # Plat the birdseye image
            plt.imshow(imgBE, cmap='gray')
            plt.show()
    
        return imgBE

#%%

from video import play_video

def findEdgeX(gray, frameid):
    sblX, sblY, sblX_thresh, sblY_thresh = findEdges(gray, thresh_rng=(20,100))    
    return sblX
#anim = play_video(findEdgeX)
##
##%%
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#
#
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format('test.mp4'))




#%%


if __name__ == '__main__':
    
    test_images = [Load(join('test_images', i)) for i in os.listdir('test_images')]
    calib_images= [Load(join( 'camera_cal', i)) for i in os.listdir('camera_cal')]
    
    test_hls = [cv2.cvtColor(im, cv2.COLOR_RGB2HLS)  for im in test_images]
    test_hsv = [cv2.cvtColor(im, cv2.COLOR_RGB2HSV)  for im in test_images]
    test_gry = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in test_images]
    
    ims1 = [np.hstack((test_gry[i], test_hls[i][:,:,0], test_hls[i][:,:,1], test_hls[i][:,:,2])) for i in range(len(test_images))]
    ims2 = [np.hstack((test_gry[i], test_hsv[i][:,:,0], test_hsv[i][:,:,1], test_hsv[i][:,:,2])) for i in range(len(test_images))]
    
    plot_hls_hsv(ims1, ims2)
    
    
    #%% Calibration
    
    ret = runCameraCalib(calib_images)
    matrix, dist = ret[0], ret[1]
    
    
    #%%
    
    imgH = np.array([hls[:,:,0] for hls in test_hls])
    imgL = np.array([hls[:,:,1] for hls in test_hls])
    imgS = np.array([hls[:,:,2] for hls in test_hls])
    
    # Undistort
    imgS_UD = np.array([undistort(img, matrix, dist)[0] for img in imgS])
    
    for img in imgS_UD:
        plt.imshow(img, cmap='gray')
        plt.show()
    
    # imgs = imgS_UD
    #
    # # Run Sobel
    # img_Sx = np.array([ ])
    
    # for img in  imgS: birdseye(img, (320, 634), (610, 438), (685, 438), (980, 634))
    # plt.imshow(imgS_UD[1])
    # plt.show()
    
    sblX, sblY, sblX_thresh, sblY_thresh = findEdges(imgS_UD[0], thresh_rng=(20,100))
    
    
    #birdseye(sblX_thresh.astype(np.uint8), (320, 634), (610, 438), (685, 438), (980, 634))
    
    #birdseye(sblX_thresh.astype(np.uint8), (295,  680), (628,  430), (647,  432), (1098, 670))
    #birdseye(sblX_thresh.astype(np.uint8), (  313.15307959,   670.75626848), (  614.1427636 ,   440.73486814), (  671.92012972,   441.42681264), ( 1103.56069178,   670.75626848))
    birdseye(sblX_thresh.astype(np.uint8), ( 279.59258069,   658.02642407), ( 594.19063035,   447.9643303 ), ( 707.82803511,   447.11194092), (1034.1251768 ,   658.02642407))
