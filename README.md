# Barnes Maze Behavioral Analysis 

This repository contains Python scripts for the behavioral analysis of rodents in the Barnes Maze task. The pipeline is divided into three main stages:

1. **Preprocessing** – Correct hole position on frames of data.

   Follow `DLC_preprocessing.ipynb`
   
3. **Analysis** – Compute behavioral metrics such as latency, path length, hole visits, and spatial strategies.

   Run `main_batch.py` and select all `.h5` files obtained with Deeplabcut. This script returns a `Final_results.h5`, `Final_results.xlsx`, with all metrics, and `Final_results_holes.h5`, `Final_results_holes.xlsx`, with the number of head dips in all holes.
   
5. **Statistics** – Perform group comparisons and statistical testing across experimental conditions.

    Run `final_statistics.py` using the `Final_results.h5` file.

