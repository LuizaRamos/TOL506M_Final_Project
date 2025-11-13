# Wildlife Image Classification with Transfer Learning and Zero-Shot Evaluation

> TÖL506M – *Introduction to Deep Learning*, University of Iceland (Fall 2025) Final Project

This project explores three complementary approaches to wildlife image classification:
1. **Training from scratch** using a convolutional neural network (ResNet-18),
2. **Fine-tuning** a pretrained ImageNet model for transfer learning,
3. **Zero-shot classification** with **SigLIP 2**, a state-of-the-art vision-language model from Google.

By comparing these techniques on the same dataset, this project highlights the trade-offs between data efficiency, computational cost, and accuracy.

---

## Objectives
- Implement and train a CNN from scratch on a wildlife dataset  
- Apply transfer learning and experiment with layer freezing strategies  
- Perform zero-shot classification with SigLIP 2  
- Generate learning curves to visualize how model performance scales with data size  
- Analyze and discuss the efficiency and generalization of each approach

---

## Dataset
The project supports multiple datasets; the experiments were run on:

**Animals-10** (Kaggle)  
- ~28 000 images across 10 animal classes  
- Moderate difficulty and balanced data distribution  
- [Dataset link](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

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
│   ├── 02_task2_finetune.ipynb
│   ├── 03_task3_zeroshot.ipynb        
│   ├── 04_task4_learning_curvers.ipynb
│   └── 05_results_summary.ipynb
│
├── results/                    # Output results, logs, and generated artifacts
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

