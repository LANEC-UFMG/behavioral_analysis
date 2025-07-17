"""
Extract Frame from Video
===============================================================================

This script contains utility functions to extract video frames and 
obtain the frame rate (FPS) from video files. It supports frame extraction
based on frame index and uses OpenCV and FFmpeg as backends for processing.

Main Sections
-------------
1. extract_frame_f_video : Extract a frame (RGB) from a specified time/frame in a video.
2. get_video_fps         : Retrieve the frame rate (FPS) of a video using FFmpeg.

Dependencies
------------
- opencv-python (cv2)
- ffmpeg-python

Author
------
Ikaro-Beraldo
Docstring comments by Rafael Bessa

Date:
"""

import cv2
import ffmpeg

def extract_frame_f_video(video_filename, video_frame=None, fps=30):
    """
    Extract a single RGB frame from a video file.

    If no specific frame index is provided, the first readable frame is returned.
    If a frame index is specified, the corresponding frame is extracted based on
    the given FPS value.

    Parameters
    ----------
    video_filename : str
        Full path to the video file (.mp4).
    video_frame : int, optional
        Index of the frame to extract. If None, the first frame is returned.
        Default is None.
    fps : float, optional
        Frame rate (frames per second) of the video. Required for calculating 
        time position when a specific frame is requested. Default is 30.

    Returns
    -------
    frame_rgb : ndarray
        The extracted video frame converted to RGB format as a NumPy array.
    """
    vidcap = cv2.VideoCapture(video_filename)  # Video capture
    ret = False
    
    # GET ONLY THE FIRST FRAME CASE THE VIDEO FRAME HAS NOT BEEN SELECTED
    if video_frame is None:
        while not ret:
            # read the first video frame
            ret,frame = vidcap.read()
        
    # IF A SPECIFIC VIDEO FRAME HAS BEEN GIVEN 
    else:
        while not ret:
            time_msec = (video_frame/fps)*1000     # Multiple by 1000 to get it in milliseconds
            vidcap.set(cv2.CAP_PROP_POS_MSEC,time_msec)      # just cue to 20 sec. position
            ret,frame = vidcap.read()
        
    # Return the frame as image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vidcap.release()
    return frame_rgb

def get_video_fps(video_filename):
    """
    Retrieve the frames-per-second (FPS) value from a video file.

    This function uses FFmpeg to extract metadata from the video file.
    It assumes the video filename is derived from a DLC filename and reconstructs
    the actual video file name accordingly.

    Parameters
    ----------
    video_filename : str
        Full path to the file used to derive the actual video filename.
        It should contain the substring 'DLC', which is used to reconstruct
        the corresponding '.mp4' video name.

    Returns
    -------
    fps : float
        Frame rate of the video (frames per second).
    """
    # Get the video filename based on the H5 DLC output name
    check_vid_filename = video_filename[0:video_filename.index('DLC')] + '.mp4'
    
    # Use ffmpeg to probe the video file
    try:
        probe = ffmpeg.probe(check_vid_filename)
    except ffmpeg.Error as e:
        print(e.stderr)

    # Create a dict to organize the video info (everything is str so, it has to be extract as int)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps1,fps2 = video_info['r_frame_rate'].split('/')
    # Get FPS value
    fps = int(fps1)/int(fps2)
    
    return fps