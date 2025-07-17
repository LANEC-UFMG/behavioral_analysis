#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select DeepLabCut Tracking Data and Video Files
===============================================================================

This script provides graphical file selection utilities for loading 
DeepLabCut (DLC) tracking data (.h5 files) and their corresponding 
video files (.mp4 format). These utilities are designed to streamline 
batch processing by returning full file paths for easy iteration in 
subsequent analysis routines.

Main Sections
-------------
1. select_file_coord : Opens a file dialog box to select one or multiple DLC tracking files (.h5)
2. select_file_video : Opens a file dialog box to select one or multiple video files (.mp4)

Dependencies
------------
- tkinter (built-in)

Author
------
Rafael Bessa  
Based on scripts (data_handling.py) by Ikaro-Beraldo (modified)

Date: October 4, 2024 
"""

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askopenfilenames

# =============================================================================    
# Function to open a simple dialog box
# =============================================================================
def select_file_coord(multiple = True):
    """
    Open a dialog box to select one or multiple DeepLabCut output (.h5) files.

    This function uses a graphical file dialog box (Tkinter) to allow the user to 
    select one or more HDF5 files containing DeepLabCut tracking data. 
    The selected file path(s) are returned as a tuple of strings.

    Parameters
    ----------
    multiple : bool, optional
        If True, enables selection of multiple files.
        If False, only a single file can be selected. Default is True.

    Returns
    -------
    filename : tuple of str
        Tuple containing the full path(s) of the selected HDF5 file(s).
        If `multiple=False`, the tuple contains a single file path.
    """
    
    # Get the filetypes
    filetypes = (('H5', '*.h5'),('All files', '*.*'))
    
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    # Check if it is multiple files
    if multiple is True:
        filename = askopenfilenames(title='Open DLC coordinate files', filetypes=filetypes)
    else:   # Unique file
        filename = askopenfilename(title='Open DLC coordinate file', filetypes=filetypes)
        print('Selected file: '+filename)
        
    # show an "Open" dialog box and return the path to the selected file    
    for i in range(len(filename)):
        filename[i].replace('/','//')   # Replace / for // because of Windows pathway default
    
    return filename

# =============================================================================
# Function to open a simple dialog box
# =============================================================================
def select_file_video(multiple = True):
    """
    Open a dialog box to select one or multiple video (.mp4) files.
    
    This function uses a graphical file dialog box (Tkinter) to allow the user to 
    select one or more MP4 video files corresponding to DeepLabCut tracking data.
    The selected file path(s) are returned as a tuple of strings.
    
    Parameters
    ----------
    multiple : bool, optional
        If True, enables selection of multiple files.
        If False, only a single file can be selected. Default is True.
    
    Returns
    -------
    filename : tuple of str
        Tuple containing the full path(s) of the selected MP4 file(s).
        If `multiple=False`, the tuple contains a single file path.
    """
    
    # Get the filetypes
    filetypes = (('MP4', '*.mp4'),('All files', '*.*'))
    
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    # Check if it is multiple files
    if multiple is True:
        filename = askopenfilenames(title='Open video files', filetypes=filetypes)
    else:   # Unique file
        filename = askopenfilename(title='Open video file', filetypes=filetypes)
        print('Selected file: '+filename)
        
    # show an "Open" dialog box and return the path to the selected file    
    for i in range(len(filename)):
        filename[i].replace('/','//')   # Replace / for // because of Windows pathway default
    
    return filename