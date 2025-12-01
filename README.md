# Wildlife Image Classification with Transfer Learning and Zero-Shot Evaluation

> TÖL506M – *Introduction to Deep Learning*, University of Iceland (Fall 2025) Final Project

This repository contains all code and experiments for comparing three computer-vision approaches to wildlife image classification using the Animals-10 dataset. The goal is to evaluate the performance, data efficiency, and computational cost of:
1. **Training from scratch** using a convolutional neural network (ResNet-18),
2. **Fine-tuning** a pre-trained ImageNet model for transfer learning,
3. **Zero-shot classification** with **SigLIP 2**, plus a linear probe based on the same model.

By comparing these techniques on the same dataset, this project highlights the trade-offs between data efficiency, computational cost, and accuracy.

---

## Objectives
- Implement and train a CNN from scratch on a wildlife dataset  
- Apply transfer learning and experiment with layer freezing strategies  
- Perform zero-shot classification with SigLIP 2  
- Generate learning curves to visualise how model performance scales with data size  
- Analyse and discuss the efficiency and generalisation of each approach

---

## Dataset
The Animals-10 [(Dataset link)](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset consists of 26,179 images across 10 classes (dog, horse, elephant, butterfly, chicken, cat, cow, sheep, spider, squirrel) as described in the report (see Table 1 on page 2).

Stratified splits used: 70% training /15% validation /15% testing.

Training subsets of 10%, 25%, 50%, 75%, and 100% were used for data-efficiency experiments.

---

## Project Structure

```
TOL506M_Final_Project/
│
├── .venv/                      # Python virtual environment
│
├── data/                       # Data handling and augmentation
│   ├── __init__.py
│   ├── augmentation.py
│   ├── dataset.py
│   └── processed/              # Data splits
│
├── models/                     # Model architectures and pretrained versions
│   ├── __init__.py
│   ├── pretrained.py
│   └── resnet_scratch.py
│
├── notebooks/                  # Jupyter notebooks (experiments, exploration)
│   ├── 00_data_exploration.ipynb
│   ├── 01_task1_scratch.ipynb
│   ├── 01_task1_scratch_2.ipynb
│   ├── 01_task1_scratch_3.ipynb
│   ├── 02_task2_finetune.ipynb
│   ├── 02_task2_finetune_2.ipynb
│   ├── 02_task2_finetune_3.ipynb
│   ├── 03_task3_zeroshot.ipynb   
│   ├── 03_task3_zeroshot_2.ipynb   
│   ├── 03_task3_zeroshot_3.ipynb        
│   └── 04_task4_learning_curvers.ipynb
│
├── results/                    # Output results
│
├── tasks/                      # Task-specific training and evaluation scripts
│   ├── __init__.py
│   ├── task1.py
│   ├── task2.py
│   └── task3.py
│
├── utils/                      # Helper functions and reusable modules
│   ├── __init__.py
│   ├── evaluation.py
│   ├── training.py
│   └── visualization.py
│
├── config.py                   # Configuration settings for the project
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

