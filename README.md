# Barnes Maze Behavioral Analysis 

This repository contains Python scripts for the behavioral analysis of rodents in the Barnes Maze task. The pipeline is divided into three main stages:

## 1. **Preprocessing** – Correct hole position on frames of data.

   Follow `DLC_preprocessing.ipynb`
   
## 2. **Analysis** – Compute behavioral metrics such as latency, speed, hole visits, and spatial strategies.

`barnes-maze-analysis folder`

Routines created in Python for Barnes Maze analysis based on DeepLabCut outputs

### a)  Download

   - Download the content of this repository
   - Click on 'Code' and 'Download Zip'


### b)  Creating a conda-environment
   - Download anaconda (https://www.anaconda.com/)
   - Open the Anaconda Prompt on Windows start
 ![image](https://github.com/ikaro-beraldo/barnes-maze-python-routines/assets/55361465/43e0eab0-567f-4abd-92dc-b9596f6a0487)
   
   - Navigate to where the file `py_routines_behavior.yaml` is, and enter:

          cd \your_path_where_is_the_file_py_routines_behavior.yaml
   
   - Then enter:
         
           conda env create -f py_routines_behavior.yaml
   
### c)  Activate the virtual environment and open Spyder
   - On Anaconda Prompt,  enter:

         conda activate py_routines_behavior
   
   - Open Spyder (Spyder is a data-science GUI IDE heavily inspired by MATLAB but for Python)

         spyder

### d)  Start running the routines
   - **For 1 trial**

The first and simplest routine is the `main_routine.py`. Run it by pressing F5 and select the DLC output file regarding the trial you want to analyze (eg. 'C38_1_G3_D4_T2DLC_resnet50_BM_GPUMar26shuffle2_700000.h5')

   - **For several trials**
  
Run `main_batch.py` and select all `.h5` files obtained with Deeplabcut. This script returns a `Final_results.h5`, `Final_results.xlsx`, with all metrics, and `Final_results_holes.h5`, `Final_results_holes.xlsx`, with the number of head dips in all holes.
   
## 3. **Statistics** – Perform group comparisons and statistical testing across experimental conditions.

   - Run `final_statistics.py` using the `Final_results.h5` file.

