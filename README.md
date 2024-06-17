# Repository: Analysis for Eco-evolutionary Guided Pathomic Analysis to Predict DCIS Upstaging

This repository contains the code and data analysis for the research paper titled "Eco-evolutionary Guided Pathomic Analysis to Predict DCIS Upstaging". The aim of this study is to predict the upstaging status of DCIS patients with eco-evolutionary biomarkers. The analysis steps are meticulously documented in the `pipeline.sh` script, which orchestrates the entire workflow from data preprocessing to result generation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Analysis Pipeline](#analysis-pipeline)
- [License](#license)

## Introduction

This study focuses on the ecological dynamics of ductal carcinoma in situ (DCIS), an early-stage breast cancer, to improve prediction of disease progression through novel biomarker identification.. This repository contains all the necessary code to reproduce the analyses presented in the paper, patholomics image data is available at [data repo]

## Installation

To run the code in this repository, you need to have the following software installed:

- Python 3.11
- R 4.0+

Clone the repository:

```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name
```

Install the required dependencies in a new conda environment:

```bash
conda create --name dcis_analysis python=3.11
conda activate dcis_analysis
pip install -r requirements.txt
```

## Analysis Pipeline

Sure! Here's a description of the `pipeline.sh` script for your GitHub README:

---

## Analysis Pipeline

The `pipeline.sh` script orchestrates the entire workflow of our analysis, from processing raw data to training the final classifier model. Below are the detailed steps included in the script:

1. **Process Raw Cell Detection of HE & IHC Output from QuPath**
   - **Script**: `process_raw_cellDetection.ipynb`
   - **Description**: This step processes the raw cell detection data from Hematoxylin and Eosin (HE) and Immunohistochemistry (IHC) images generated by QuPath.

2. **Co-register HE with IHCs Using VALIS for the Same Patients**
   - **Script**: `warpPoints_valis.ipynb`
   - **Description**: This step aligns HE images with corresponding IHC images for the same patients using the VALIS software to ensure accurate spatial registration.

3. **Define the Niches**
   - **Script**: `./cc.sh`
   - **Description**: This step involves defining ecological niches within the tumor microenvironment based on the processed and registered images.

4. **Extract the Features**
   - **Scripts**:
     - `computeMor_text.py`
     - `process_spatialFunc.ipynb`
     - `aggregate_and_split.ipynb`
   - **Description**: In this multi-part step, various features are extracted from the images. This includes computing morphological features, processing spatial functions, and aggregating the data before splitting it for further analysis.

5. **Train the Classifier Model for Regions of Interest (ROIs) of Different Scales**
   - **Script**: `classifier_fromCluster_multiscale.py`
   - **Description**: The final step involves training a classifier model on the extracted features, considering ROIs of different scales to predict the upstaging of DCIS.

Each step is documented and the code is modularized for ease of understanding and modification.

## License

This project is licensed under the [MIT License](LICENSE).

---

If you have any questions or need further information, please feel free to contact Yujie Xiao at yujie.xiao@stonybrook.edu.

---

Thank you for your interest in our work!

Yujie Xiao

Stony Brook University

---


