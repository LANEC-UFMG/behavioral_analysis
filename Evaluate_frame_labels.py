#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Frame Labels – Tracking Confidence Assessment and Correction
===============================================================================

This script processes DeepLabCut tracking data to assess and correct 
tracking confidence for body part labels in video frames. It allows:

- Automatic detection of trial onset/offset based on likelihood.
- Quantification of low-confidence frames for each body part.
- Manual correction of body part positions in low-confidence frames.
- Visualization and documentation of tracking quality.
- Manual or automated labeling of fixed spatial zones (e.g., escape holes).

Main Sections
-------------
1. get_trial_onset_offset  : Estimate start/end of trial based on label likelihood.
2. check_frames_likelihood : Quantify label reliability and generate summary plots.
3. fix_frames_likelihood   : Manually correct low-confidence frames and interpolate.
4. define_hole_position    : Annotate spatial zones either manually or from DLC data.

Dependencies
------------
- numpy
- pandas
- scipy
- matplotlib
- opencv-python (cv2)
- extract_frame / mousecollect_func (custom)

Author
------
Rafael Bessa  
Based on functions (get_trial_beginnning_end_all_bp() and fix_frames_confidence()
from the script data_processing.py) by Ikaro-Beraldo (modified)

Date: July 8, 2025
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from mousecollect_func import * # Script to import define_bodyparts() and define_hole_position()
import cv2
from matplotlib import pyplot as plt
import gc

# =============================================================================
# Function: get_trial_onset_offset
# =============================================================================
def get_trial_onset_offset(df, conf_threshold, fps=30, max_duration_trial=3, n_bp=9):
    """
    Estimate the onset and offset of a trial based on body part label likelihood.

    This function searches for the first and last frames where at least a percentage
    of body parts exceed a given confidence threshold, indicating a valid trial.
    
    Parameters
    ----------
    df : pandas.DataFrame
        MultiIndex dataframe with DLC tracking output.
    conf_threshold : float
        Minimum likelihood required for a body part to be considered correctly detected.
    fps : int, optional
        Frames per second of the video (default is 30).
    max_duration_trial : int, optional
        Maximum trial duration in minutes (default is 3).
    n_bp : int, optional
        Number of body parts tracked (default is 9).
    
    Returns
    -------
    beg : int
        Index of the first frame of the trial.
    end : int
        Index of the last frame of the trial.
    """
    
    bps = df.columns.get_level_values(1)    
    
    # Get unique bp values
    indexes = np.unique(bps, return_index=True)[1]  # Get unique
    [bps[index] for index in sorted(indexes)]       # Resort them
    
    # Pre-allocate 
    bpt_cf = np.zeros((len(df),n_bp))
    
    for ii in range(n_bp):  # Loop for each body part
        bpt_cf[:,ii] = df.xs(bps[ii], level='bodyparts', axis=1).to_numpy()[:,2]  # Get confidence interval for each pb

    r_mask, c_mask = np.where(bpt_cf >= conf_threshold)       # Find the frames and bps where confidence interval is above the threshold
    unique, counts = np.unique(r_mask, return_counts=True)    # Get the number of counts
    
    # Define the trial beginning as the first moment when (80% of the animals bp have high confidence interval)
    beg = unique[np.where(counts >= round(n_bp*0,8))[0][0]]
    # Define the trial end as the last moment when (80% of the animals bp have high confidence interval)
    end = unique[np.where(counts >= round(n_bp*0,8))[0][-1]]
    
    # Check if the trial lasted more than the max limit defined by the user
    if (end-beg+1)/fps > max_duration_trial*60:
        end = beg + max_duration_trial*60*fps
          

    return beg, end
       
# =============================================================================
# Function: check_frames_likelihood
# =============================================================================
def check_frames_likelihood(video_list, coord_list, conf_threshold, n_bp, savepath, plt_format='png'):
    """
    Evaluate tracking confidence across body parts and generate visual and tabular summaries.

    This function plots frame snapshots at onset and offset (middle columns), as well as
    snapshots before and after onset/offse (time interval set to 3 seconds). The function
    also plots likelihood curves and pie charts for each body part, and compiles the
    statistics into a DataFrame.
    
    Parameters
    ----------
    video_list : tuple of str
        List of paths to video files.
    coord_list : tuple of str
        List of paths to DeepLabCut HDF5 coordinate files.
    conf_threshold : float
        Threshold for determining low vs high likelihood.
    n_bp : int
        Number of body parts.
    savepath : str
        Directory to save generated plots.
    plt_format : str, optional
        File format for saving figures (default is 'png').
    
    Returns
    -------
    video_info : pandas.DataFrame
        DataFrame summarizing likelihood stats for each labeled body part and trial boundaries per video.
    """

    for i in range(len(coord_list)):
        
        df = pd.read_hdf(coord_list[i]) # Load df coords
        
        # ---------------------------------------------------------------------
        # Create info dataframe
        # ---------------------------------------------------------------------
        if i == 0:            
            colnames = df.columns.get_level_values(1).unique().tolist()
            bp_list = colnames[:n_bp]
            cols = ['Filename','FPS','Frame_on','Frame_off','confidence_thre']
            
            for ii in range(n_bp):
                           cols.append(bp_list[ii] + '_lowconf_nframes')
                           cols.append(bp_list[ii] + '_lowconf_percent')
            
            video_info = pd.DataFrame(data=None,columns = cols, index = range(len(video_list)))
            # path = video_list[i][0:55]
            
        if len(video_list[i]) == 65:
            fname = video_list[i][-18:-4]
        else:
            fname = video_list[i][-22:-4]
        
        # ---------------------------------------------------------------------
        # Load video info   
        # ---------------------------------------------------------------------
        cap = cv2.VideoCapture(video_list[i])
        fps = cap.get(cv2.CAP_PROP_FPS)
        # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get frame onset and offset
        t_onset, t_offset = get_trial_onset_offset(df, conf_threshold, fps=round(fps), max_duration_trial=3, n_bp=n_bp)
        # Array for Onset/Offset, pre and post frames
        ti = 3 # time interval pre and post frames onset/offset (in seconds)
        f = np.array([round(-fps)*ti, 0, round(fps)*ti])        
        
        # ---------------------------------------------------------------------
        # Plot frames Onset/Offset
        # ---------------------------------------------------------------------
        figg, axs = plt.subplots(2,3,sharey=True)               
        frame_id = np.zeros(6,dtype=int)
    
        for m in range(3):
            if (m == 0) & (t_onset+f[0] < 0):
                frame_id[m] = 0
                frame_id[m+3] = t_offset+f[m]
                
            elif (m == 2) & (t_offset+f[2] >= frame_count):
                frame_id[m] = t_onset+f[m]
                frame_id[m+3] = frame_count-1
                
            else:
                frame_id[m] = t_onset+f[m]
                frame_id[m+3] = t_offset+f[m]
            
           
        for mm in range(6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id[mm])
            ret,frame = cap.read()
            if ret:
                axs.flat[mm].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                axs.flat[mm].set_title(f'Frame {frame_id[mm]}')
            else:                                      
                print(f"Aviso: Não foi possível ler o frame {frame_id[mm]} do vídeo {fname}")

        
        cap.release()
        plt.savefig(savepath + fname + '_Frame_OnOff', format=plt_format)
        plt.close(figg)
        
        # ---------------------------------------------------------------------
        # Load infos to dataframe
        # ---------------------------------------------------------------------
        video_info.loc[i,'Filename'] = video_list[i]
        video_info.loc[i,'FPS'] = round(fps)
        video_info.loc[i,'Frame_on'] = t_onset
        video_info.loc[i,'Frame_off'] = t_offset
        video_info.loc[i,'confidence_thre'] = conf_threshold
        
        df_crop = df.iloc[t_onset:t_offset+1,:] # Crop df by trial onset and offset
        
        # ---------------------------------------------------------------------
        # Plot and write info of body parts likelihood per frames
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(11, 7),
                                         gridspec_kw={'width_ratios': [2, 1, 2, 1, 2, 1],
                                                      'height_ratios': [1, 1, 1], 'wspace': 0.2,'hspace': 0.6})
                           
        piecolors = ['#FA8072' , '#2D82B5']
    
        for j in range(0,len(bp_list)):  
            
            # calculate the proportion of good/bad frames using confidence likelihood
            label_likelihood = df_crop.xs((bp_list[j], 'likelihood'),
                                          level=('bodyparts', 'coords'), axis=1).to_numpy() 
            low_conf = label_likelihood <= conf_threshold
            low_total = low_conf.sum() # count number of 'True's
            frame_total = len(df_crop)
            
            # load bp likelihood info to dataframe
            video_info.loc[i,bp_list[j] + '_lowconf_nframes'] = low_total
            video_info.loc[i,bp_list[j] + '_lowconf_percent'] = (low_total/frame_total) * 100
            
            # plot  
            ax = plt.subplot(3,6,(j+1)+(j*1))        
            plt.plot(label_likelihood, color = 'gray')
            ax.axhline(conf_threshold, linestyle='--', color='r')
            ax.set_title(bp_list[j])
            
            if (j == 0) | (j == 3) | (j == 6):
                ax.set_ylabel("Likelihood")
                
            if j>5:      
                ax.set_xlabel("Frame idx")
                
            plt.subplot(3,6,(j+2)+(j*1)) 
            plt.pie([low_total, frame_total-low_total], autopct='%1.1f%%', colors = piecolors)
           
            if j==2:
                plt.legend(['Low likelihood','High likelihood' ], fontsize=8,
                           bbox_to_anchor=(-0.5, 2), loc='upper left', borderaxespad=0.)
                          
        plt.savefig(savepath + fname + '_BPLikelihood', format=plt_format)
        plt.close(fig)
        
        del df, df_crop
        print(i)
        gc.collect()
            
    return video_info

# =============================================================================
# Function: fix_frames_likelihood
# =============================================================================
def fix_frames_likelihood(video_list, coord_list, video_info, conf_threshold, bp_list,
                          seq_thre, frame_jump, savepath, id_on=None):
    """
    Identify and correct sequences of low-confidence frames via manual annotation and interpolation.

    This function detects frame sequences where tracking confidence falls below threshold, allows 
    manual correction of body part coordinates, and interpolates across the time series.
    
    Parameters
    ----------
    video_list : tuple of str
        List of full paths to video files.
    coord_list : tuple of str
        List of full paths to DLC output files.
    video_info : pandas.DataFrame
        DataFrame with metadata including trial onset and offset.
    conf_threshold : float
        Minimum likelihood required for a valid body part label.
    bp_list : list of str
        Names of body parts to consider.
    seq_thre : int
        Minimum number of consecutive low-confidence frames to trigger correction.
    frame_jump : int
        Frame sampling interval for manual annotation.
    savepath : str
        Directory to save output files.
    id_on : int, optional
        Offset index to match video_info with video_list (default is 0).
    
    Returns
    -------
    None
        Saves corrected coordinates and metadata as .npy files.
    """
    
    id_on = 0 if id_on is None else id_on 
    for i in range(len(coord_list)):
        print(i)
        # Load initial inputs
        df = pd.read_hdf(coord_list[i])
        vname = video_list[i]
        t_onset = video_info['Frame_on'][i+id_on]
        t_offset = video_info['Frame_off'][i+id_on]
        n_frames = df.shape[0]
        n_coords = len(df.columns.get_level_values('coords').unique())
        n_bodyparts = len(bp_list)
        
        if len(video_list[i]) == 65:
            fname = video_list[i][-18:-4]
        else:
            fname = video_list[i][-22:-4]
        
        # Crop the data to get only the desired body parts
        df.columns = df.columns.droplevel('scorer')
        coords_name = df.columns.get_level_values('coords').unique().tolist()
        filtcol = [(bp, c) for bp in bp_list for c in coords_name]    
        df_crop = df[filtcol] 
        body_part_matrix = df_crop.to_numpy().reshape(n_frames, n_bodyparts, n_coords)
        
        confidence_mask = np.zeros(np.shape(body_part_matrix)[0:2], dtype=bool)    
        
        # Create a vector with low confidence frames from all body parts
        for i in range(n_bodyparts):    
            # Get a confidence mask based on the confidence threshold
            confidence_mask[:,i] = body_part_matrix[:,i,2] >= conf_threshold
            
        conf_mask = np.all(confidence_mask == True, axis=1)
     
        exc_frames = np.transpose(np.asarray(np.where(conf_mask == False)))   # Coords excluded (<0.95)
        acc_frames = np.transpose(np.asarray(np.where(conf_mask == True)))  # Coords accepted (>0.95)
        
        # ---------------------------------------------------------------------
        # Detect low confidence frames sequence edges for all body parts
        # ---------------------------------------------------------------------
        diff_exc = np.diff(exc_frames,axis=0)           
        edge = np.where(diff_exc > 1)[0][:] # Borda
        sequence_on = exc_frames[edge+1] # Start frames of long sequences with low confidence
        sequence_off = exc_frames[edge] # End frames of long sequences with low confidence
        
        if diff_exc[0] == 1:           
            sequence_on = np.sort(np.append(sequence_on,exc_frames[0]))
            
        else:
            sequence_off = exc_frames[edge][1:]
            
        if diff_exc[-1] == 1:
            sequence_off = np.append(sequence_off,exc_frames[-1])
            
        else:
            sequence_off = np.append(sequence_off,exc_frames[edge[-1]+1])
                    
        sequences_all = np.stack((sequence_on,sequence_off),axis=1).reshape(len(sequence_on),2)
        
        # Correct onset/offset cases
        onset_idx = np.where((sequences_all[:,0]<t_onset) & (sequences_all[:,1]>=t_onset))[0]
        offset_idx = np.where((sequences_all[:,0]<=t_offset) & (sequences_all[:,1]>t_offset))[0]
        sequence_on[onset_idx] = t_onset
        sequence_off[offset_idx] = t_offset
        
        len_sequences = sequence_off-sequence_on # vector with length of all sequences and its size       
        sequences_cut = sequences_all[len_sequences > seq_thre].copy()
        less_onset = np.any(sequences_cut<t_onset,axis=1)
        high_offset = np.any(sequences_cut>t_offset,axis=1)   
        sequences = np.delete(sequences_cut,(less_onset|high_offset),axis=0) # vector with long sequences to correct
       
        # ---------------------------------------------------------------------
        # Manual correction
        # ---------------------------------------------------------------------
        new_coords, corrected_frames = define_bodyparts(body_part_matrix, n_bodyparts, sequences, vname, frame_jump)
        
        # ---------------------------------------------------------------------
        # Correct old coords with manually defined coords
        # Make nan all coords with low conf
        # ---------------------------------------------------------------------
        body_part_matrix[exc_frames,:,0:2] = np.nan
        
        # Substitute old coords with new coords
        body_part_matrix[corrected_frames,:,0:2] = np.asarray(new_coords)
        
        # Interpolate 
        acc_frames = np.sort(np.append(acc_frames,corrected_frames))
        frame_seq = np.arange(np.shape(body_part_matrix)[0])
        bp_matrix_interp = body_part_matrix.copy()
        
        for j in range(n_bodyparts):
            for jj in range(2):
                # Extract temporal series for each body part coordinate (x and y)
                coordinate = body_part_matrix[:, j, jj]
    
                if np.sum(acc_frames) < 2:
                    # Not enough points to interpolate
                    continue
    
                # Interpolator
                interp_func = interp1d(frame_seq[acc_frames], 
                    coordinate[acc_frames], kind='linear', bounds_error=False,
                    fill_value="extrapolate"  # or use None to leave as nan out of limits
                )
    
                # Interpolate all frames
                bp_matrix_interp[:, j, jj] = interp_func(frame_seq)
         
        # Return interpolated coords matrix and respective info
        kdict = ['bodyparts_list','likelihood_thre','sequence_thre','frame_jump',
                 'excluded_frames','frame_sequences_all','frame_sequences_corrected','corrected_frames']
        vdict = [bp_list, conf_threshold, seq_thre, frame_jump,
                 exc_frames, sequences_all, sequences, corrected_frames]
        correction_data = dict(zip(kdict,vdict)) # Dictionary type
        
        np.save(savepath+fname+'_processedInfo.npy', correction_data, allow_pickle=True)
        np.save(savepath+fname+'_processedCoords.npy', bp_matrix_interp)
        
        # Break from loop (optional)
        if not i==(len(coord_list)-1):
            aswr = input("Press Enter to continue or 'q' to exit: ")
    
        if aswr.lower() == 'q':
            print(f"You finished file {fname} (iteration {i})")
            break
        
    return

# =============================================================================
# Function: define_hole_position
# =============================================================================
def define_hole_position(video_list, coord_list, video_info, nholes, n_bp, savepath,
                         frame_idx, old_hole_coords=None, recheck_id=None, dv=0):
    """
    Retrieve or manually define coordinates of fixed spatial references (e.g., escape holes).

    Allows user to accept DeepLabCut-detected positions or manually click each hole on a frame.
    Results are saved as images and stored in a summary table.
    
    Parameters
    ----------
    video_list : tuple of str
        List of video file full paths.
    coord_list : tuple of str
        List of DLC output file (HDF5) full paths.
    video_info : pandas.DataFrame
        DataFrame containing onset frame indices and filenames.
    nholes : int
        Number of spatial zones (e.g., holes) to be labeled.
    n_bp : int
        Number of body parts used in tracking.
    savepath : str
        Output directory to save results.
    frame_idx : int
        Frame index to be selected
    old_hole_coords : pandas.DataFrame, optional
        Previous dataframe containing hole coordinates (default is None)
    recheck_id : numpy.ndarray, optional
        Vector containing the indexes to be reviewed. Indexes refer to old_hole_coords
        dataframe (default is None)
    dv : int, optional
        If 0, uses DLC coordinates; if 1, collects them manually (default is 0).
    Returns
    -------
    hole_coords : pandas.DataFrame
        DataFrame containing hole coordinates for each video.
    """
    
    # -------------------------------------------------------------------------
    # Check optional parameters
    # -------------------------------------------------------------------------
    if old_hole_coords is None:
          
        # ---------------------------------------------------------------------
        # Create output dataframe
        # ---------------------------------------------------------------------        
        cols = ["FileID"]
        hname = [f"h{i:02d}_{sufixo}" for i in range(1,nholes+1) for sufixo in ['x', 'y']]
        cols.extend(hname)        
        hole_coords = pd.DataFrame(data=None,columns = cols, index = range(len(video_info)))        
        iteration_vec = range(len(video_list))

    else:
        # ---------------------------------------------------------------------
        # Use optional variables
        # ---------------------------------------------------------------------       
        hole_coords = old_hole_coords
        iteration_vec = recheck_id-1
                        
    # -------------------------------------------------------------------------
    # Start loop through coord list
    # -------------------------------------------------------------------------
    for i in iteration_vec:
        
        onset = video_info['Frame_on'][i]+frame_int
        
        # color vector
        colors = ['red']*nholes
        colors[0] = 'green'        
                  
        # FileID
        if len(video_info['Filename'][i]) == 65:
            hole_coords.loc[i,'FileID'] = video_info['Filename'][i][-18:-4]
        else:
            hole_coords.loc[i,'FileID'] = video_info['Filename'][i][-22:-4]
        
        # Get frame object            
        cap = cv2.VideoCapture(video_list[i])        
        cap.set(cv2.CAP_PROP_POS_FRAMES, onset)
        ret,frame = cap.read()
        if ret:
            frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:                                      
            print(f"Aviso: Não foi possível ler o frame {onset} do vídeo {hole_coords.iloc[i,0]}")
        
        cap.release()
        
        # ---------------------------------------------------------------------
        # Get DLC holes coords
        # ---------------------------------------------------------------------
        if dv==0: # Accept DLC label
            df = pd.read_hdf(coord_list[i]) # Load df coords
            dfcopy = df.copy()
            colname = dfcopy.columns.get_level_values('coords').unique()[0:2] # Sub-level coords' columns names ('x' and 'y')
            cf = [col for col in dfcopy.columns if col[2] in colname] # List comprehension
            hc = dfcopy[cf[n_bp*2:]] # Holes x y columns
            
            hole_coords.iloc[i,1:] = hc.iloc[onset,:]
            hcoords = hole_coords.iloc[i,1:].values.reshape(-1, 2)                      
            
        else: # Manually define labels
            hcoords = define_hole(frame_img, nholes)
            hole_coords.iloc[i,1:] = hcoords.reshape(np.size(hcoords))
        
        # Plot frame and holes
        fig,ax = plt.subplots()
        ax.cla() # clear things for fresh plot
        ax.imshow(frame_img)
        ax.set_title('Holes positions')
        ax.scatter(hcoords[:, 0], hcoords[:, 1], color=colors, s=40)
        plt.savefig(savepath + hole_coords.iloc[i,0] + '_HolePosition', format='png')
        plt.close(fig)
        
        print(i)
 
    return hole_coords