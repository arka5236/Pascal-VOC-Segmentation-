# Pascal-VOC-Segmentation-
Design, train, and evaluate a computationally lightweight, end-to-end deep learning model using PyTorch for semantic segmentation on the PASCAL VOC 2012 dataset. The goal is to maximize accuracy (Dice Score) while minimizing inference-time complexity (FLOPs), achieving a balance suitable for real-world deployment.

# Dataset & TaskDataset: PASCAL VOC 2012 (21 classes, including background).
#Input: RGB Image (3, 300, 300).Output: Pixel-wise segmentation mask (300, 300) with integer class labels [0-20].

# Training/Validation Split: 
80:20 ratio from the official training set. Strict Rule: The original VOC 2012 validation set must not be used for training or hyperparameter tuning.
# Key RequirementsComputationally Lightweight: 
Focus on low FLOPs for rapid inference.
# Robustness:
The model must maintain performance on corrupted images (e.g., Gaussian noise, salt-and-pepper noise, blur, compression artifacts). Augmenting training data with noisy versions is encouraged.
# End-to-End: 
Direct mapping from input to mask without post-processing (e.g., CRF).
# Framework:
Exclusively PyTorch.

# Evaluation MetricsAccuracy: 
Dice Similarity Coefficient (DSC) computed as a macro-average over all 21 classes.

# Efficiency:
Floating-Point Operations (FLOPs) at inference time.
