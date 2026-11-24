## Research Paper

## Introduction
This project explores industrial anomaly detection using Robust Distribution Alignment (RoDA). It uses defect-free product images to build a normal-feature memory bank, then scores test images by comparing their local visual features with that bank. Experiments evaluate adaptation and preprocessing under image corruption, and few-shot classification of defect types.

## Project contents
- RoDa.ipynb — Main RoDA implementation: feature extraction, normal memory-bank construction, Sinkhorn-based adaptation, defect classification, and few-shot experiments. Its final section evaluates pill and screw defect types using New_defectdata.
- RoDA+Normalization.ipynb — Evaluates per-image min–max normalization followed by ImageNet normalization.
- RoDA+Normalization+SD.ipynb — Evaluates contrast-shift detection and correction, with ROC and recall plots.
- RoDA_Normalizer+GaussianDenoise.ipynb — Compares Gaussian smoothing and non-local means (NLM) under Gaussian noise.
- RoDA+Normalization+Weiner.ipynb — Contains normalization and preprocessing experiments with Gaussian-noise AUROC and recall reporting.
- RoDa_EfficientNetB0.ipynb — An alternate backbone experiment using EfficientNet-B0.
- wiener_improvements.py — Helper functions for Wiener deblurring and recall evaluation.
- data/ — Main MVTec-AD dataset.

## Dataset Layout
<img width="535" height="510" alt="image" src="https://github.com/user-attachments/assets/81bb9386-6630-4e85-97a8-ec363e6d8cfb" />


## Environmental Setup
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python pillow imagecorruptions scikit-image openpyxl jupyter

## Running the experiments
1. Open the notebook for the experiment you want to run.
2. Check the dataset root and category settings.
3. Restart the kernel and run cells in order.
4. Run the experiment and result-export cells for the plots and tables.
The main RoDA experiments use a pretrained Wide ResNet-50-2 and features from layer2 and layer3. The default input processing resizes images to \(256 \times 256\), center-crops to \(224 \times 224\), and applies ImageNet normalization. The normalization notebook also applies per-image min–max scaling.

For corruption experiments, synthetic corruption is applied to target images before the target data is split into adaptation and evaluation subsets. In the common RoDA configuration, 20% is used for adaptation. Corruption type, severity, coreset ratio, and adaptation settings are defined in each notebook and may be overridden in later experiment cells.
New-defect few-shot experiment
The RoDa.ipynb few-shot section uses three labeled support images per defect type, with remaining images reserved as queries. It compares frozen-feature prototypes with a low-rank few-shot head; the feature-extractor backbone remains frozen. The New_defectdata evaluation applies no synthetic corruption.
Outputs and caches
Notebooks save results as CSV, Excel, or NPZ files, depending on the experiment. Output names are specified in the relevant notebook cells. For example, the new-defect evaluation writes New_defectdata_fewshot_results.csv.

