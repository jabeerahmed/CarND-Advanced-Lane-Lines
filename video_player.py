#importing some useful packages
#import matplotlib.pyplot as plt
import numpy as np
import cv2
#import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray    

    
def process_video(filename):
    clip1 = VideoFileClip(filename)
    clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    return clip    


def play_video(filename):
    HTML("""
    <video width="1280" height="720" controls>
      <source src="{0}">
    </video>
    """.format(filename)) 
   


def save_video(clip, filename):
    clip.write_videofile(filename, audio=False)

#%%

#clip = process_video('project_video.mp4')
#save_video(clip, 'test.mp4')
play_video('test.mp4')
