# earthquake-spectrograms-classifier

## Models Used for Earthquake Spectrogram Classification

Primary goal: application of 3 different models for small dataset image classification with severe class imbalance.

This project investigates the classification of spectrogram images derived from seismic signals. The classification task is multi-class with a small number of labeled samples per class (10–100 per class), including two visually similar and difficult-to-distinguish categories. The data is balancing on the edge of ML realm thus massive augmentation was needed (mirroring images, copying left/right halves). Not all augmentation techniques were available (like noise adding) because of data specific features.  The following three modeling strategies were developed and evaluated, each model actually a cascade of two-step models for distinguishing hard and easy separable classes:

### 1. **Convolutional Neural Network (CNN) Baseline**

The first model is a baseline supervised classifier for spectrogram image classification, based on transfer learning with the ResNet-18 architecture. The model uses a pretrained ResNet-18 network, with the final classification layer replaced to output predictions for four custom classes: brak, noise_clean, noise_ice_cracking, and noise_lake.

All input images are resized to 224x224 pixels and normalized using ImageNet mean and standard deviation, to match the requirements of ResNet-18. Images are loaded from train, validation, and test folders using PyTorch’s ImageFolder utility. Original dataset class labels are mapped to a reduced set of four target classes. The mapping is consistently applied to all data splits using a custom dataset wrapper.

The model is trained using cross-entropy loss and the Adam optimizer with a learning rate of 1e-4. Batch size and number of training epochs are configurable; typical values are a batch size of 16 and 10 training epochs. The training pipeline uses data loaders with shuffling enabled for the training set and deterministic ordering for validation and testing. Training is performed on a GPU if available.

This setup provides a straightforward benchmark for the spectrogram classification problem, allowing for direct comparison with more advanced architectures and techniques. The pipeline ensures clear class definitions, reproducible data splits, and compatibility with further experiments using transformer-based or prompt-tuned models.



### 2. **Vision Transformer with Visual Prompt Tuning (VPT) on DeiT-Small/16**

2. Vision Transformer with Visual Prompt Tuning (VPT) on DeiT-Small/16
This project leverages DeiT-Small/16, a data-efficient Vision Transformer (ViT) model, enhanced with Visual Prompt Tuning (VPT) for few-shot spectrogram classification under severe class imbalance.

Key Techniques and Implementation
Model Base:
The backbone is DeiT-Small/16, a lightweight Vision Transformer pretrained on ImageNet. The original weights of DeiT are kept frozen to preserve their feature extraction capabilities.

Prompt Tuning:
Visual Prompt Tuning (VPT) is implemented in two variants. VPT-Shallow: Learnable prompt tokens are introduced only at the input layer of the transformer. VPT-Deep: Learnable prompt tokens are injected into the input of each transformer block. In both variants, only the prompt tokens and the final classification head are trainable; all other transformer parameters remain frozen, significantly reducing the number of trainable parameters.

Class Head Replacement: the standard classification head of DeiT is replaced with a new trainable layer corresponding to the four target classes.

Data Preprocessing: input spectrograms are resized and center-cropped to 224×224 pixels, normalized using ImageNet statistics. Three data splits are used (train, val, test), loaded from directory structure. All labels are directly mapped (no relabeling or merging).

Class Imbalance Handling: the dataset is highly imbalanced across four classes: brak_Empty zone, brak_Station, noise_DNS, noise_Tip-off. Per-class weights are computed as w_i = N / (K * n_i) for use in the loss function, where n_i is the number of samples in class i, N is the total sample count, and K is the number of classes.

Custom Loss Function: training uses Focal Loss with class weights to mitigate class imbalance and improve learning on rare classes.

Training and Hyperparameter Optimization: Optuna is used for automated hyperparameter search, optimizing parameters such as learning rate, weight decay, batch size, prompt length, focal loss gamma, and VPT variant (shallow vs. deep). Early stopping is applied based on macro-averaged F1 score on the validation set.

Scheduler: learning rate scheduling uses cosine decay with linear warmup for improved convergence.

---

### 3. **PCA + HOG + SVM cascade**

The third model family uses a classical machine learning approach based on Histogram of Oriented Gradients (HOG) feature extraction, Principal Component Analysis (PCA) for dimensionality reduction, and Support Vector Machine (SVM) classification with an RBF kernel. This approach is applied in two variants for different classification subtasks.

First, all input spectrogram images are resized to 128x128 pixels and converted to grayscale. HOG descriptors are computed for each image to produce compact and informative feature vectors. For the three-class subtask, the original four classes are mapped as follows: "brak_Empty zone" and "brak_Station" are mapped to separate classes, while "noise_DNS" and "noise_Tip-off" are merged into a single combined noise class. For the two-class subtask, only the noise-related classes are retained: "noise_DNS" and "noise_Tip-off", mapped to two separate classes.

Datasets are constructed by applying the class mapping to all images and optionally filtering to retain only the relevant classes. HOG features are extracted for each image in the training, validation, and test splits. Before model training, all feature vectors are standardized using a z-score scaler fitted on the training data.

To further reduce dimensionality and noise, PCA is applied after scaling. The retained variance fraction for PCA is treated as a hyperparameter and tuned during cross-validation. SVM classifiers are trained on the resulting PCA-transformed features, with balanced class weights to account for class imbalance. The SVM's regularization parameter (C), RBF kernel width (gamma), and PCA variance threshold are all optimized using Optuna with stratified 5-fold cross-validation and macro-averaged F1 score as the target metric.

Model selection is performed on the validation split. The best models are then used for final performance evaluation, including calculation of classification metrics and confusion matrices. This classical pipeline provides both a robust baseline and a modular framework for structured spectrogram classification, and can be easily adapted to different class mappings or feature extraction strategies as needed.

---

Each model was evaluated using metrics such as per-class accuracy, macro-averaged F1 score, and confusion matrix analysis to assess its performance, especially on visually similar class pairs. This three-model comparison highlights the challenges of working with limited labeled spectrogram data and the benefits of modern transformer-based techniques in such scenarios.

---

