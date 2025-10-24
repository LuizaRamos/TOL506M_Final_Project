# Wildlife Image Classification with Transfer Learning and Zero-Shot Evaluation

> TÖL506M – *Introduction to Deep Learning*, University of Iceland (Fall 2025) Final Project

This project explores three complementary approaches to wildlife image classification:
1. **Training from scratch** using a convolutional neural network (ResNet-18),
2. **Fine-tuning** a pretrained ImageNet model for transfer learning,
3. **Zero-shot classification** with **SigLIP 2**, a state-of-the-art vision-language model from Google.

By comparing these techniques on the same dataset, this project highlights the trade-offs between data efficiency, computational cost, and accuracy.

---

## 🎯 Objectives
- Implement and train a CNN from scratch on a wildlife dataset  
- Apply transfer learning and experiment with layer freezing strategies  
- Perform zero-shot classification with SigLIP 2  
- Generate learning curves to visualize how model performance scales with data size  
- Analyze and discuss the efficiency and generalization of each approach

---

## 🦒 Dataset
The project supports multiple datasets; the experiments were run on:

**Animals-10** (Kaggle)  
- ~28 000 images across 10 animal classes  
- Moderate difficulty and balanced data distribution  
- [Dataset link](https://www.kaggle.com/datasets/alessiocorrado99/animals10)


...

