#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Coordinate Collection Script for Video Frames
===============================================================================

This script allows for the manual and interactive collection of coordinates for body parts
or specific points (such as escape holes) in frames extracted from videos, using
matplotlib for visualization and clicking. It can be used to correct automatic tracking
or manually annotate points of interest in animal experiments.

Main Sections
-------------
1. define_bodyparts : Collects body part coordinates across sequences of frames.
2. define_hole      : Collects coordinates of holes (or zones) from a single frame.

Dependencies
------------
- matplotlib
- numpy
- extract_frame (function extract_frame_f_video)
- opencv-python (cv2; indirectly, via extract_frame)

Author
-----
Rafael Bessa

Date: May 23, 2025
"""

from matplotlib import pyplot as plt
import numpy as np
from extract_frame import * # Script to import extract_frame_f_video()

# =============================================================================
# define_bodyparts group of functions
# =============================================================================

# -----------------------------------------------------------------------------
# Class: ClickCollector
# -----------------------------------------------------------------------------
class ClickCollector:
    """
    Interactive click collector for annotating body parts in a frame.
    
    Allows manual clicks to record coordinates, with support for undoing and 
    skipping clicks. Use mouse's 3 buttons:
        
    - Left button: add point
    - Middle button: undo last click
    - Right button: skip (insert NaN)
    
    Parameters
    ----------
    
    coords : np.ndarray
        Previous coordinates to be overlaid on the frame as a reference (shape: [n, 2]).
    ls : int
        Total number of frame sequences.
    lf : int
        Total number of frames in the current sequence.
    count1 : int
        Index of the current sequence.
    count2 : int
        Index of the current frame.
    n_clicks : int
        Number of points to be clicked.
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax : matplotlib.axes.Axes
        Matplotlib axis where clicks are recorded.
    """
    
    def __init__(self, coords, ls, lf, count1, count2, n_clicks, fig, ax):
        self.n_clicks = n_clicks
        self.fig = fig
        self.coords = coords
        self.ax = ax
        self.clicks = []
        self.patches = []
        self.count1 = count1
        self.count2 = count2
        self.ls = ls
        self.lf = lf
        self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.done = False

    def onclick(self, event):
        """
        Handles mouse click events on the matplotlib axis.
        """
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left button: add point
            self.clicks.append((x, y))
            self.ax.plot(x, y, 'ro', markersize=3)
            self.fig.canvas.draw()
            
        elif event.button == 3: # Right button: skip
            self.clicks.append((np.nan,np.nan))
            
        elif event.button == 2 and self.clicks:  # Middle button: undo
            self.clicks.pop()
            self.ax.clear()            
            self.ax.imshow(self.current_frame)
            self.ax.set_title(f'Sequence {self.count1}/{self.ls}; Frame {self.count2}/{self.lf}')
            
            for i in range(self.coords.shape[0]):
                circle = plt.Circle((self.coords[i,0], self.coords[i,1]), 2, color='b', alpha=0.5) #h_1 -> h_12 clocwise
                self.ax.add_patch(circle) 
                            
            for cx, cy in self.clicks:
                self.ax.plot(cx, cy, 'ro', markersize=3)

            self.fig.canvas.draw()

        if len(self.clicks) == self.n_clicks:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.done = True
            plt.close()

    def collect(self, frame_img):
        """
        Displays the frame and collects the user's clicks.
        
        Parameters
        ----------
        frame_img : np.ndarray
            Frame image to be annotated
        
        Returns
        -------
        list of tuple
            List of clicked (x,y) coordinates              
        """
        self.current_frame = frame_img
        self.ax.imshow(frame_img)
        self.ax.set_title(f'Sequence {self.count1}/{self.ls}; Frame {self.count2}/{self.lf}')
        self.done = False
        plt.show(block=True)
        return self.clicks       

# -----------------------------------------------------------------------------
# Function: define_bodyparts
# -----------------------------------------------------------------------------
def define_bodyparts(body_part_matrix, n, sequence, video, fr_int):
    """
    Interactively collects body part coordinates from selected video frames.
    
    Parameters
    ----------
    body_part_matrix : np.ndarray
        3D matrix (frames x body_parts x [x, y]) with previously estimated coordinates.
    n : int
        Number of points (body parts) to be clicked per frame.
    sequence : np.ndarray
        Matrix (n_seq x 2) with (start, end) frame pairs for each sequence.
    video : str
        Path to the video.
    fr_int : int
        Interval between frames to be annotated.
    
    Returns
    -------
    all_coords : list of list of tuple
        List containing the clicked coordinates for each frame.
    frame_idx : np.ndarray
        Indices of the frames used for annotation.
    """    
    all_coords = []
    frame_idx = []
    ls = len(sequence) # length sequence
    count1 = 0

    for i in range(ls):
        frame_indices = np.arange(sequence[i, 0] + 4, sequence[i, 1], fr_int)
        frame_idx = np.concatenate((frame_idx,frame_indices),axis=0)
        count1 += 1
        
        count2 = 0

        for j in range(len(frame_indices)):
            frame_img = extract_frame_f_video(video, video_frame=frame_indices[j], fps=30)
                        
            fig, ax = plt.subplots(1,1,figsize=(19.2,10.8))
            old_coords = body_part_matrix[frame_indices[j],:,0:2] # 3D to 2D array
            ax = plt.gca()
            ax.cla() # clear things for fresh plot
            count2 += 1
            lf = len(frame_indices) # length frame_indices
            
            for ii in range(0,body_part_matrix.shape[1]):
                circle = plt.Circle((old_coords[ii,0],old_coords[ii,1]), 2, color='b', alpha=0.5) #h_1 > h_12 sentido horário
                ax.add_patch(circle)
                           
            collector = ClickCollector(old_coords, ls, lf, count1=count1, count2=count2, n_clicks=n, fig=fig, ax=ax)
            coords = collector.collect(frame_img)
            all_coords.append(coords)
            
    frame_idx = np.int64(frame_idx)

    return all_coords, frame_idx

# =============================================================================
# define_hole group of functions
# =============================================================================

# ------------------------------------------------------------------------------
# Classe: ClickCollector2
# ------------------------------------------------------------------------------
class ClickCollector2:
    """
    Interactive click collector for marking holes (or fixed zones).

    Parameters
    ----------
    n_clicks : int
        Number of points to be clicked.
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    ax : matplotlib.axes.Axes
        Matplotlib axis.
    """
    def __init__(self, n_clicks, fig, ax):
        self.n_clicks = n_clicks
        self.fig = fig
        self.ax = ax
        self.clicks = []
        self.patches = []
        self.cid = fig.canvas.mpl_connect('button_press_event', self.onclick2)
        self.done = False

    def onclick2(self, event):
        """
        Handles mouse click events on the matplotlib axis.
        """
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left button: add point
            self.clicks.append((x, y))
            self.ax.plot(x, y, 'ro', markersize=4)
            self.fig.canvas.draw()
            
        elif event.button == 3 and self.clicks:  # Right button: undo
            self.clicks.pop()
            self.ax.clear()            
            self.ax.imshow(self.current_frame)
            self.ax.set_title('Start from escape hole. Keep labeling clockwise')
                                       
            for cx, cy in self.clicks:
                self.ax.plot(cx, cy, 'ro', markersize=4)

            self.fig.canvas.draw()

        if len(self.clicks) == self.n_clicks:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.done = True
            plt.close()

    def collect2(self, frame_img):
        """
        Displays the frame and collects the user's clicks.
        
        Parameters
        ----------
        frame_img : np.ndarray
            Frame image to be annotated.
        
        Returns
        -------
        list of tuple
            List of clicked (x, y) coordinates.
        """
        self.current_frame = frame_img
        self.ax.imshow(frame_img)
        self.ax.set_title('Start from escape hole. Keep labeling clockwise')
        self.done = False
        plt.show(block=True)
        return self.clicks       


# -----------------------------------------------------------------------------
# Function: define_hole
# -----------------------------------------------------------------------------
def define_hole(frame_img, nholes):
    """
    Manually collects the coordinates of the holes (or points of interest) in a frame.
    
    Parameters
    ----------
    frame_img : np.ndarray
        Input image.
    nholes : int
        Total number of holes (or points) to be clicked.
    
    Returns
    -------
    coords : np.ndarray
        Matrix (nholes x 2) containing the (x, y) coordinates of the clicked holes.
    """    
                    
    fig, ax = plt.subplots()
    ax.cla() # clear things for fresh plot
    ax.imshow(frame_img)
    ax.set_title('Start from escape hole. Keep labeling clockwise')
               
    collector = ClickCollector2(nholes, fig, ax)
    coords = collector.collect2(frame_img)
    coords = np.array(coords)

    return coords