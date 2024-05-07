# GAIT (Gesture Analysis and Interpretation Toolkit)

Welcome to the GAIT (Gesture Analysis and Interpretation Toolkit) repository for EI (Entornos Inteligentes) subject! This repository includes all scripts written for processing and analysing different marching pathrons. In addition to pure gait classification pipelines, this repository also includes the analysis of psychological traits of the person being filmed using [Psymo dataset](https://openaccess.thecvf.com/content/WACV2024/papers/Cosma_PsyMo_A_Dataset_for_Estimating_Self-Reported_Psychological_Traits_From_Gait_WACV_2024_paper.pdf).

## Features

- **Gesture Preprocessing**:
  - Normalize and preprocess raw sensor data.
  - Extract key features from gesture signals.

- **Gesture Representation**:
  - Convert raw sensor data into gesture representations suitable for analysis.

- **Gesture Analysis**:
  - Implement algorithms for gesture segmentation and recognition.
  - Test the training results on non-seen image sequences.

## Installation

To use this project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/DanielFrauAlfaro/gait.git
   
2. **Installation**:
   ```
   pip3 install torch torchvision      # or your preferred Pytorch version
   sudo apt-get install -y python3-opencv
   pip3 install numpy==1.24
   pip3 install utils
   pip3 install tqdm
   pip3 install termcolor
   pip3 install kornia
   pip3 install einops
   pip3 install -U scikit-learn
3. **Usage**:
   ```bash
   python3 main.py
