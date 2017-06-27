# import IPython
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 1')
# get_ipython().magic('aimport main')
# get_ipython().magic('aimport video')
# get_ipython().magic('matplotlib inline')



# %load_ext autoreload
# %autoreload 1
# %aimport main
# %aimport video
# %matplotlib inline

import os
from random import randrange, sample
from os.path import join
from skimage import io, color
from matplotlib.pyplot import imshow, plot
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


#%% Import the required modules
# get_ipython().magic('pylab inline')

# import cv2
# from IPython.display import clear_output
import imageio

def noop(frame): return frame

def run_video(filename='project_video.mp4', fheight=5, func=noop, start=0, nframes=1000000000, cmap=None, **kwargs):
    # Grab the input device, in this case the webcam
    # You can also give path to the video file
    vid = imageio.get_reader(filename)
    num_frames = vid.get_length()
    nframes = min(num_frames, nframes)
    start = max(0, start)
    stop  = min(start + nframes, start + num_frames)
    
    def getFigSize(frame, fheight=5):
        w,h = frame.shape[1], frame.shape[0]
        aspect = float(w)/float(h)
        fw, fh = (fheight*aspect, fheight) if (aspect > 1) else (fheight, float(fheight)/aspect)
        return (fw,fh)
        
    fig, axs = None, None
    
    # Put the code in try-except statements
    # Catch the keyboard exception and 
    # release the camera device and 
    # continue with the rest of code.
    try:
        
        for i in range(start,stop):
            # Capture frame-by-frame
            frame = vid.get_data(i)
            # Convert the image from OpenCV BGR format to matplotlib RGB format
            # to display the image
            frame = func(frame, **kwargs)
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=getFigSize(frame, fheight))                
            # Turn off the axis
            axis('off')
            # Title of the window
            title("Frame Number: {} of {}".format(i, nframes))
            # Display the frame
            imshow(frame, cmap=cmap, aspect='equal')
            show()
            # Display the frame until new frame is available
            clear_output(wait=True)
    except KeyboardInterrupt:
        # Release the Video Device
        vid.close()
        # Message to be displayed after releasing the device
        print("Released Video Resource")


#%% ## Road Trapezoid
def create_region_mask(imshape, vertices, ignore_mask_color = 255):
    """ Returns a mask given the image's shape, vertices [array].
        Defaults:
        ignore_mask_color = 255
    """
    mask = np.zeros((imshape[0], imshape[1]), dtype=np.uint8)        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask == ignore_mask_color


def road_trapezoid(ind=-1):
    coords = [
                [(320, 634), (610, 438), (685, 438), ( 980, 634)],
                [(295, 680), (628, 430), (647, 432), (1098, 670)],
                [(313, 670), (614, 440), (671, 441), (1103, 670)],
                [(279, 658), (594, 447), (707, 447), (1034, 658)],
                [(162, 674), (608, 433), (690, 435), (1120, 665)],
                [(190, 679), (559, 449), (714, 442), (1098, 670)]
             ]
    
    return coords[ind]
road_trapezoid()


#%% ## BirdsEye

from skimage.transform import matrix_transform as transform

def transformImage(img, matrix):
    return cv2.warpPerspective(img, matrix, (img.shape[:2])[::-1], flags=cv2.INTER_LINEAR)


def transformPoints(pts, matrix):
    return transform(pts, matrix)


def poly2Lines(pts):
    npts = len(pts)
    return np.array([[pts[i], pts[(i+1) % npts]] for i in range(npts)]).reshape(npts*2,2)    


def undistort(img, mtx, dist, retRoi=True):
    imsize = (img.shape[:2])[::-1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist, imsize,1, imsize)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)    
    x,y,w,h = roi
    ret = (dst, dst[y:y+h, x:x+w]) if retRoi else dst
    return ret


class BirdsEye(object):
    
    
    def CalcDstCoords(botL, topL, topR, botR):
        d_botL = (  botL[0], 650)
        d_botR = (  botR[0], 650)
        d_topL = (d_botL[0], 50)
        d_topR = (d_botR[0], 50)
        return d_botL, d_topL, d_topR, d_botR

    def GetTransform(botL, topL, topR, botR):
        src = np.array((botL, topL, topR, botR));
        # Get the ideal destination coords
        dst = np.array(BirdsEye.CalcDstCoords(botL, topL, topR, botR));
        # Forward transform matrix
        matrix  = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        # Inverse Transform matrix
        inv_mat = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
        return src, dst, matrix, inv_mat #np.matrix(matrix), np.matrix(inv_mat)
    
    def TransformImage(image, pts=((164, 669), (607, 435), (667, 433), (1160, 657)), retMats=False, retPts=False):
        # Calculate Transformation matrices
        src, dst, mat, inv_mat = BirdsEye.GetTransform(*tuple(pts))
        # Transform Image
        ret = transformImage(image, mat)
        if retMats: ret = (ret, mat, inv_mat)
        if retPts : ret = ret + (src, dst)
        return ret


#%% ## Color Space Conversion

def clrCvt(im, flag, ch):
    ch = min(2, ch)
    im2 = cv2.cvtColor(im, flag)
    return im2 if (ch < 0) else im2[:,:,ch]
    
def hls(im, ch=2):
    return clrCvt(im, cv2.COLOR_RGB2HLS, ch)

def hsv(im, ch=1):
    return clrCvt(im, cv2.COLOR_RGB2HSV, ch)


#%% ## Colorspace Analysis

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


#%% ## Camera Calibration 

class CamCalib:

    def calcCalibParams(objpoints, imgpoints, imgs):
        imsize = (imgs[0].shape[:2])[::-1]
        ret = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)
        ret, mtx, dist, rvecs, tvecs = ret
        return mtx, dist

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

    def Run(calib_images, draw=False):
        # Find Corners
        objpoints, imgpoints, imgs = CamCalib.FindCorners(calib_images)    
        # Calculate Calib Matrix
        matrix, dist = CamCalib.calcCalibParams(objpoints, imgpoints, imgs)
        # Undistort Images
        undist_imgs = np.array([undistort(img, matrix, dist) for img in imgs])
        undist_imgs, undist_roi_imgs = undist_imgs[:,0], undist_imgs[:,1]

        if (draw):
            # Plot some images        
            n_plot = 5
            grid_imgs = [np.hstack((imgs[i], undist_imgs[i])) for i in sample(range(len(imgs)), n_plot)]
            plot_grid(5,1, lambda r, c, i, fig, axs: axs[r,c].imshow(grid_imgs[i]))

        return matrix, dist, objpoints, imgpoints, imgs, undist_imgs, undist_roi_imgs    


#%% ## Sobel / Edge Detection


class Sobel:
    def Dir_thresh(gray, sobel_kernel=9, thresh=(-180, 180)):
        # Convert to grayscale
        if len(gray.shape) == 3: gray = hls(gray)
        # Convert to radians if needed
        if (thresh[1] > np.pi/2): thresh = np.deg2rad(thresh)

        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir).astype(bool)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output 
    
    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def Mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100)):
        # Convert to grayscale
        if len(gray.shape) == 3: gray = hls(gray)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary magnitude and direction thresholds
        mask_mag = (gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])
        # Create a binary image
        binary_output = np.zeros_like(gradmag)
        binary_output[mask_mag] = 1
        # Return the binary image
        return binary_output

class uda:

    def find_centers(warped, window_width=80, margin=100, ):
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(1*warped.shape[0]/4):int(3*warped.shape[0]/4),:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(1*warped.shape[0]/4):int(3*warped.shape[0]/4),int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        return l_center, r_center
        
    def find_window(warped, window_width, window_height, margin):

        def window_mask(width, height, img_ref, center,level):
            output = np.zeros_like(img_ref)
            output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
            return output

        def find_window_centroids(warped, window_width, window_height, margin):

            window_centroids = [] # Store the (left,right) window centroid positions per level
            window = np.ones(window_width) # Create our window template that we will use for convolutions

            # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
            # and then np.convolve the vertical image slice with the window template 

            # Sum quarter bottom of image to get slice, could use a different ratio
            l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
            r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
            r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

            # Add what we found for the first layer
            window_centroids.append((l_center,r_center))

            # Go through each layer looking for max pixel locations
            for level in range(1,(int)(warped.shape[0]/window_height)):
                # convolve the window into the vertical slice of the image
                image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
                conv_signal = np.convolve(window, image_layer)
                # Find the best left centroid by using past left center as a reference
                # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
                offset = window_width/2
                l_min_index = int(max(l_center+offset-margin,0))
                l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
                # Find the best right centroid by using past right center as a reference
                r_min_index = int(max(r_center+offset-margin,0))
                r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
                # Add what we found for that layer
                window_centroids.append((l_center,r_center))

            return window_centroids

        window_centroids = find_window_centroids(warped, window_width, window_height, margin)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        return output


#%% ## Image / Rect Utilities

def widen(roi, left=0, right=0):
    pts = np.array(roi)
    pts[:2, 0] -= left
    pts[2:, 0] += right
    return pts

def shorten(roi, top=0, bot=0):
    pts = np.array(roi)
    pts[0, 1] += bot
    pts[3, 1] += bot
    pts[1, 1] += top
    pts[2, 1] += top    
    return pts

def reshape_hwc(img):
    return img if (len(img.shape)==3) else img.reshape(img.shape + (1,))

def gray3(gray): return np.stack((gray,)*3, axis=2)

def getColor(img):
    return img if (len(img.shape)==3) else gray3(img)

def polylines(im, pts, color=[0,200,0], thickness=5, isClosed=True, copy=True):
    im = getColor(np.copy(im) if copy else im)
    return cv2.polylines(im, pts=np.array([pts], dtype=np.int32), isClosed=isClosed, color=color, thickness=thickness)
    
def polyfills(im, pts, color=[0,200,0], copy=True):
    im = getColor(np.copy(im) if copy else im)
    return cv2.fillPoly(im, pts=np.array([pts], dtype=np.int32), color=color)

def circle(im, pts, radius=10, color=[0,200,0], thickness=-1, copy=True):
    im = getColor(np.copy(im) if copy else im)
    for pt in pts: cv2.circle(im, tuple(pt), radius=radius, color=color, thickness=thickness)
    return im
    
def rect(im, topLeft, sizeWH, color=[0,200,0], thickness=5, fill=False, copy=True):
    x, y = topLeft
    w, h = sizeWH
    pts  = np.array([(x, y), (x+w,y), (x+w, y+h), (x,y+h)], dtype=np.int)
    poly = polyfills if fill else polylines
    return poly(im, pts, color=color, thickness=thickness, copy=copy, isClosed=True)
        
def apply_mask(img, mask, false_value=0):
    dst = np.copy(reshape_hwc(img))
    if not (type(false_value)==list or type(false_value)==np.ndarray): 
        false_value = [false_value]*dst.shape[2]
    for i in range(dst.shape[2]):
        dst[mask == False, i] = false_value[i]            
    return dst.reshape(img.shape)

# In[18]:

from scipy.misc import imresize
from scipy.signal import convolve2d, convolve, gaussian


def renderAllImages(frame, ud_frame=None, be_img=None, be_sbl=None, src=None, dst=None, be_mask_pts=None):
    def _imshow(im, name="Image", cmap=None):
        plt.title(name), plt.xticks([]), plt.yticks([])
        plt.imshow(im, cmap=cmap), plt.show()
    
    ## Show Original
    _imshow(frame, "Original")

    ## Undistort
    if ud_frame is not None: _imshow(ud_frame, "Undistorted")

    ## Apply Bird's Eye Transformation
    if (be_img is not None) and (src is not None) and (dst is not None):            
        ud_poly, be_img_poly = polylines(ud_frame*255, src), polylines(be_img*255, dst)
        _imshow(np.hstack((ud_poly, be_img_poly)), "Bird's Eye Transform")
                    
        ## After Sobel-Magnitude-Thresholding
        if be_sbl is not None:
            _imshow(be_sbl, "Sobel-Mask", cmap='gray')
            _imshow(apply_mask(be_img, be_sbl), "Sobel-Mask Applied to Bird's Eye View")
            

def doAll(frame, sbl_param = (7, (25, 150)), be_pts=[], show_plots=False, verbose=False):    
        
    def create_curve(poly_coeff, h, ploty=None, skip=10):
        '''Given a quadratic polynomial, calculate'''
        if len(poly_coeff) == 0: return []
        if ploty is None: ploty = np.arange(0, h).reshape(h,1)[::skip]
        curve = poly_coeff[0]*ploty**2 + poly_coeff[1]*ploty + poly_coeff[2]
        return [[int(c), int(p)] for c,p in zip(curve, ploty)] #np.hstack((curve, ploty)).astype(int)

    def fit_lane_to_curves(pts):
        # Check validity
        if (len(pts) < 2): return []
        # Convert to numpy array
        pts = np.array(pts,dtype=np.int)
        # Fit pts to 2nd order poly
        lane_fit = np.polyfit(pts[:,1], pts[:,0], 2)
        # Check error
        return [] if (len(lane_fit) != 3) else list(lane_fit)
    
    def find_peaks(patch, center, offset=0, width=None, axs=None):
        n = len(patch)
        if width is None: width = int(n * 0.4)
        n_min, n_max = max(center - int(width/2), 0), min(center + int(width/2), n)
        window = patch[n_min:n_max]
        pmean, pstd = window.mean(), window.std()
        p = np.nonzero(window >= (pmean + pstd))[0]
        pts = []
        if (len(p) == 0):
            if verbose: print("len(p) == 0")
        elif (pmean < 1.0):
            if verbose: print("pmean < 1.0")
        else:
            x = int(np.mean(p))
            y = window[x]
            pts = [x + offset + n_min, y] if (y > 0) else []
            if verbose: print("x, y = {:3.1f}, {:3.1f}  |  Pts = {:3.1f}, {:3.1f}".format(x, y, pts[0], pts[1]))
                
        if show_plots:
            print("[min, center ,max] = [{}, {}, {}],  mean|std = {:2.5f} | {:2.5f}, Pmean|Pstd = {:2.5f} | {:2.5f}".format(n_min, center, n_max, pmean, pstd, patch.mean(), patch.std()))
            axs.plot(patch)
            axs.axvspan(n_min, n_max, color='g', fill=False)
            if (len(pts)>0): axs.plot(x + n_min, y, 'rx')
        return pts, [int(n_min + offset), int(n_max+offset)]
    
    def find_left_right_peaks(patch, centers):
        patch = np.copy(patch)
        patch[patch < patch.mean()] = 0
        for i in range(int(len(patch)/2), len(patch)):
            n = i
            if patch[i] == 0: break
        if n == (len(patch)-1): return [],[], n
        fig, axs = plt.subplots(1, 2, sharey=True) if show_plots else (None, [None]*2)
        left, l_winx = find_peaks(patch[:n], centers[0], axs=axs[0])
        right,r_winx =find_peaks(patch[n:], centers[1]-n, axs=axs[1], offset=n)        
        if show_plots: plt.show()
        return left, right, n, l_winx, r_winx

    def find_lane_markers(be_sbl, num_seqs=16, show_plots=False):
        h, w = be_sbl.shape
        step = int(h/num_seqs)
            
        norm = lambda x: x/np.linalg.norm(x)
        ksize = 31
        filt = cv2.getDerivKernels(1, 0, ksize)
        div, box = norm(filt[0].reshape(ksize)), norm(gaussian(ksize, ksize/3))
        kernel = div * box
        
        im = None
        left_pts, right_pts, mid_pts = [], [], []
        skip = 4
        l_center, r_center = uda.find_centers(be_sbl)
        centers = []
        
        downsample = lambda a, s=3: a[:s*(len(a)//s)].reshape((len(a)//s), s).sum(axis=1)
        
        for i in range(h, 0, -step):
            patch = be_sbl[(i-step):i, :]
            patch_sum = patch.sum(axis=0)
            if skip > 1: patch_sum = downsample(patch_sum, skip)
            patch_dif = np.convolve(patch_sum, kernel, mode='same')
            patch_sum = np.convolve(patch_sum, box,    mode='same')
            
            if len(centers) == 0: centers = np.array([(l_center, frame.shape[0]*(7.0/8.0)), (r_center, frame.shape[0]*(7.0/8.0))], dtype=np.int)
            p_centers = (np.array(centers,)[:,0] / skip).astype(np.int)
            left, right, n, l_winx, r_winx = find_left_right_peaks(patch_sum, p_centers)
            l_winx = np.array(l_winx, dtype=np.int) * skip
            r_winx = np.array(r_winx, dtype=np.int) * skip

            y_pt = int(np.mean([i, i-step]))
            if (len(left) > 0): 
                left   = [int(left[0]*skip), y_pt]
                left_pts.append(left)
                centers[0] = left
            if (len(right) > 0): 
                right  = [int(right[0]*skip), y_pt]
                right_pts.append(right)
                centers[1] = right

            if im is None: im = getColor(be_sbl*255)
            im = rect(im, (0,i-step), (w, step), copy=False)
            im = rect(im, (l_winx[0],i-step), (l_winx[1] - l_winx[0], step), color=[0,255,255], copy=False)
            im = rect(im, (r_winx[0],i-step), (r_winx[1] - r_winx[0], step), color=[0,255,255], copy=False)
            im = circle(im,  left_pts, color=[255,0,0], copy=False)
            im = circle(im, right_pts, color=[255,0,0], copy=False)

            if show_plots: 
                plt.imshow(im)
                plt.show()
                                
        # Fit a second order polynomial to each
        curve_l, curve_r = fit_lane_to_curves(left_pts), fit_lane_to_curves(right_pts)
        curve_pts_l, curve_pts_r = create_curve(curve_l, h), create_curve(curve_r, h)

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img = circle(img, curve_pts_l, radius=12, color=[0,  255, 80], copy=False)
        img = circle(img, curve_pts_r, radius=12, color=[255,250, 50], copy=False)
        img = circle(img, left_pts,    radius=10, color=[255, 0, 0],   copy=False)
        img = circle(img, right_pts,   radius=10, color=[0, 255, 80],  copy=False)  
        img = circle(img, centers,     radius=10, color=[0,255,255],   copy=False)        
        
        points, curves = (left_pts+right_pts), (curve_pts_l+[p for p in reversed(curve_pts_r)])
        return img, im, points, curves, [curve_l, curve_r]

    blend = lambda imgA, imgB, a=0.5, b=0.5, dtype=np.uint8: (a*imgA + b*imgB).astype(dtype)
    
    ## Undistort
    ud_frame = undistort(frame, calib_matrix, calib_dist, retRoi=False)    

    ## Apply Bird's Eye Transformation
    be_img, mat, inv_mat, src, dst = BirdsEye.TransformImage(ud_frame, pts=be_pts, retMats=True, retPts=True)
    
    ## After Sobel-Magnitude-Thresholding
    be_sbl = Sobel.Mag_thresh(be_img, *sbl_param)
    
    ## Make a mask that covers the destination quad
    be_mask_pts = widen(dst, left=50, right=150)
    be_mask_pts = shorten(be_mask_pts, top=-50, bot=-30)
    be_mask = create_region_mask(be_sbl.shape, np.array([be_mask_pts], dtype=np.int32)).astype(bool)
    be_sbl[be_mask == False] = 0
    
    ## Show images until now
    if show_plots: renderAllImages(frame, ud_frame, be_img, be_sbl, src, dst, be_mask_pts)    
        
    ## Find lane lines
    res = find_lane_markers(be_sbl, show_plots=show_plots)
    img, tmp_img, points, curves, coeffs = res
    
    ## Project the output image on to the road
    img_masked    = apply_mask(img, be_mask, false_value=[170, 0, 100])
    img_projected = transformImage(img_masked, inv_mat)

    ## Project Points on undistorted frame
    points = transformPoints(points, inv_mat).astype(np.int)
    curves = transformPoints(curves, inv_mat).astype(np.int)
    ud_img = np.zeros_like(ud_frame)
    ud_img = polyfills(ud_img, curves, color=[0,  255, 80, 0], copy=False)
    ud_img = circle(ud_img,   points, radius=4, color=[255, 0,   0], copy=False)
    ud_img = blend(ud_frame, ud_img, a=0.7, b=0.3)

    be_sbl_lines = blend(img, getColor(be_sbl*255))
    
    im1, im2, im3 = be_sbl_lines, tmp_img, ud_img
    return np.vstack((np.hstack((imresize(im1, 0.5), imresize(im2, 0.5))), im3))

#################################### RUN ####################################

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML

if __name__ == '__main__':
    #%% ## Load Images
    test_images = [io.imread(join('test_images', i)) for i in os.listdir('test_images')]
    calib_images= [io.imread(join( 'camera_cal', i)) for i in os.listdir('camera_cal')]

    #%% ## Color Analysis
    test_hls = [cv2.cvtColor(im, cv2.COLOR_RGB2HLS)  for im in test_images]
    test_hsv = [cv2.cvtColor(im, cv2.COLOR_RGB2HSV)  for im in test_images]
    test_gry = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in test_images]
    ims1 = [np.hstack((test_gry[i], test_hls[i][:,:,0], test_hls[i][:,:,1], test_hls[i][:,:,2])) for i in range(len(test_images))]
    ims2 = [np.hstack((test_gry[i], test_hsv[i][:,:,0], test_hsv[i][:,:,1], test_hsv[i][:,:,2])) for i in range(len(test_images))]
    plot_hls_hsv(ims1, ims2)

    ## Calibrate
    ret = CamCalib.Run(calib_images)
    calib_matrix, calib_dist = ret[0], ret[1]

    _Pts = road_trapezoid(3)
    sbl_param=(15, (10, 255))

    def process(frame): return doAll(frame, be_pts=_Pts, sbl_param=sbl_param)


    def process_video(filename='project_video.mp4'):
        clip1 = VideoFileClip(filename)
        clip = clip1.fl_image(process) #NOTE: this function expects color images!!
        return clip    

    def save_video(clip, filename):
        clip.write_videofile(filename, audio=False)

    clip = process_video()
    save_video(clip, "my_test.mp4")


    # run_video(start=620, fheight=10, func=doAll, be_pts=_Pts, sbl_param=sbl_param, cmap='gray')



def process(frame):
    return doAll(frame, be_pts=_Pts, sbl_param=sbl_param)


    # test_frame = vid.get_data(626)
    # get_ipython().magic('time img = doAll(test_frame, be_pts=_Pts, show_plots=True, sbl_param=sbl_param, verbose=True);')
    # plt.imshow(img), plt.show;
