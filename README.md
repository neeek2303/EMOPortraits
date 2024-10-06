# EMOPortraits

Official implementation of **EMOPortraits: Emotion-enhanced Multimodal One-shot Head Avatars**.

![EMOPortraits Example](./data/EP_v.gif)

## Overview

EMOPortraits introduces a novel approach for generating realistic and expressive one-shot head avatars driven by multimodal inputs, including extreme and asymmetric emotions.

For more details, please refer to:
- [Project Page](https://neeek2303.github.io/EMOPortraits/)
- [Research Paper](https://arxiv.org/abs/2404.19110)
- [FEED Dataset Repository](https://github.com/neeek2303/FEED)
- [Main author's Page](https://neeek2303.github.io)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Environment Setup](#1-environment-setup)
  - [Installing Additional Repositories](#2-installing-additional-repositories)
  - [Downloading Pre-trained Models and Dependencies](#3-downloading-pre-trained-models-and-dependencies)
- [Usage](#usage)
- [Notes](#notes)
  - [Note 1: Repository Maintenance](#note-1-repository-maintenance)
  - [Note 2: Importance of High-Quality Data](#note-2-importance-of-high-quality-data)
  - [Note 3: Utilizing the FEED Dataset](#note-3-utilizing-the-feed-dataset)
  - [Note 4: Model Weights](#note-4-available-model-weights)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Installation

### 1. Environment Setup

You can set up the environment using the provided `conda-pack` archive:

#### Option 1: Use `conda-pack` Archive

1. **Download the Environment Archive**

   Download `sav.tar.gz` from the [Google Drive](https://drive.google.com/drive/folders/1xo_ceslle1kckpFhUS5j1-ZnYDVlToVL) or [Yandex Disk](https://disk.yandex.com/d/vq-3-wlLXlc1yw).

2. **Unpack the Environment**

   ```bash
   # Create a directory for the environment
   mkdir -p sav_env
   
   # Unpack the tar.gz archive into the directory
   tar -xzf sav.tar.gz -C sav_env
   ```

3. **Using Python Without Activating**

   ```bash
   # Run Python directly from the unpacked environment
   ./sav_env/bin/python
   ```

4. **Activating the Environment**

   ```bash
   # Activate the environment
   source sav_env/bin/activate
   ```

   Once activated, you can run Python as usual:

   ```bash
   (sav_env) $ python
   ```

5. **Cleanup Prefixes**

   After activating the environment, you may need to run the following command to fix any issues with environment paths:

   ```bash
   (sav_env) $ conda-unpack
   ```

   This command can also be run without activating the environment, as long as Python is installed on the machine.

#### Option 2: Using `environment.yml`

   *Note: This option may not work as it has not been thoroughly tested.*

### 2. Installing Additional Repositories

Due to limitations with `conda-pack`, the following repositories need to be installed manually:

1. **Face Detection**: Install from [GitHub](https://github.com/hhj1897/face_detection) 

   ```bash
   git clone https://github.com/hhj1897/face_detection.git
   cd face_detection
   pip install -e .
   ```

2. **ROI Tanh Warping**: Install from [GitHub](https://github.com/ibug-group/roi_tanh_warping)

   ```bash
   git clone https://github.com/ibug-group/roi_tanh_warping.git
   cd roi_tanh_warping
   pip install -e .
   ```

3. **Face Parsing**: Install from [GitHub](https://github.com/hhj1897/face_parsing)

   ```bash
   git clone https://github.com/hhj1897/face_parsing.git
   cd face_parsing
   pip install -e .
   ```

### 3. Downloading Pre-trained Models and Dependencies

1. **Download Required Files**

   Please download the following files from [Google Drive](https://drive.google.com/drive/folders/1xo_ceslle1kckpFhUS5j1-ZnYDVlToVL?usp=sharing) or [Yandex Disk](https://disk.yandex.com/d/vq-3-wlLXlc1yw):

   - `logs.zip` (contains main model weights - not yet available)
   - `logs_s2.zip` (contains stage 2 model weights)
   - `repos.zip` (contains dependencies repos and it's weights)

2. **Extract Files**

   Extract all the downloaded zip files into the root directory of the project:

   ```bash
   unzip logs.zip -d ./
   unzip logs_s2.zip -d ./
   unzip repos.zip -d ./
   ```

3. **Download and Extract Loss Models**

   Navigate to the `losses` directory and download the following files:

   ```bash
   cd losses
   ```

   - `loss_model_weights.zip`
   - `gaze_models.zip`

   Extract them within the same `losses` directory:

   ```bash
   unzip loss_model_weights.zip -d ./
   unzip gaze_models.zip -d ./
   ```

---

## Usage

*Instructions on how to run the code, train models, and perform inference will be added here.*

---

## Notes

### Note 1: Repository Maintenance

This repository is primarily intended for demonstration purposes, allowing enthusiasts to explore the network architecture and training procedures in detail. The primary author is not currently affiliated with academia and may not have the capacity to actively maintain this repository. Community contributions and support are highly encouraged.

### Note 2: Importance of High-Quality Data

A significant factor contributing to the success and quality of the results is the dataset used for training. The original model was trained on a high-quality (HQ) version of the **VoxCeleb2** dataset, which is no longer publicly available. However, there are now newer datasets of higher quality and larger scale. Utilizing these can potentially yield even better results, as seen in recent methods that build upon ideas presented in the **MegaPortraits** paper.

### Note 3: Utilizing the FEED Dataset

Our **FEED** dataset ([link](https://github.com/neeek2303/FEED)), introduced in our paper, was instrumental in incorporating asymmetric and extreme emotions into the latent emotion space. We encourage the community to actively use and expand upon this dataset. Given that the final version is slightly smaller (due to some participants withdrawing consent), supplementing it with other datasets containing extreme emotions (e.g., **NeRSemble**) can enhance model performance, especially when attempting to replicate or improve upon the techniques presented in **EMOPortraits**.

### Note 4: Model Weights

We are providing version of the pre-trained model weights (located in logs.zip):

1. **Retrain_with_17_V1_New_rand_MM_SEC_4_drop_02_stm_10_CV_05_1_1**

This model will be retrained using the same parameters as described in our paper but with 17 IDs in the **FEED** dataset instead of the original 23. Since the **FEED** dataset samples were used 25% of the time during training, this change might slightly affect performance in intensive tests.

Please refer to **notebooks/E_emo_infer_video.ipynb**

---

## Acknowledgements

We extend our gratitude to all contributors and participants who made this project possible. Special thanks to the developers of the datasets and tools that were instrumental in our research.

---

## License

This project is licensed under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You are free to use, modify, and distribute this work non-commercially, as long as appropriate credit is given and any derivative works are licensed under identical terms.
