# Hate Speech Detection Model

## Problem Statement
Hate speech refers to any communication that attacks or uses derogatory language against individuals or groups based on religion, ethnicity, nationality, race, color, ancestry, sex, or identity factors. This project focuses on developing a Machine Learning model to detect hate speech in tweets using Python.

## Business Objective
To create a classifier that identifies and flags hate speech on Twitter, ensuring a safer and more inclusive online community. This tool can be used by:
- **Social Media Platforms**: Automatically detect and remove hate speech to improve user experience.
- **Businesses**: Protect reputation by maintaining respectful online communication.

---

## Project Lifecycle

| **Phase**                  | **Deadline** | **Description**                                                                 |
|----------------------------|--------------|---------------------------------------------------------------------------------|
| **Problem Understanding**  | 24 March     | Analyze the problem and set objectives.                                        |
| **Data Cleaning**          | 29 March     | Preprocess Twitter data: cleaning, normalization, and transformation.          |
| **Representation Learning**| 5 April      | Use transformers (e.g., BERT, RoBERTa) to extract contextual word embeddings.   |
| **Model Training**         | 12 April     | Build and optimize the hate speech classifier using embeddings and fine-tuning.|

---

## Approach

### 1. **Exploratory Data Analysis (EDA)**
- **Dataset**: Twitter data with ~32K training samples and ~17K test samples.
- **Label Distribution**: Highly imbalanced (90% non-hate speech, 10% hate speech).
- **Visualization**: Generated word clouds for both hate and non-hate speech content.

### 2. **Data Preprocessing**
- Text tokenization using `spacy` and `nltk`.
- Removal of punctuation, stopwords, and numbers.
- Stemming and text normalization.

### 3. **Representation Learning**
- Built vocabulary using `torchtext`, ignoring words with low frequency.
- Used pre-trained embeddings (Word2Vec) and saved them for visualization using [TensorFlow Projector](https://projector.tensorflow.org).

### 4. **Model Architecture**
- Custom Transformer-based model implemented in PyTorch:
  - **Embedding Layers**: Word and positional embeddings.
  - **Transformer Encoder**: Captures contextual relationships in the text.
  - **Dense Layers**: Fully connected layers for classification.
- Loss: Binary Cross-Entropy (BCE).
- Optimizer: Adagrad (handles infrequent word updates efficiently).

### 5. **Training and Validation**
- Batch size: 128.
- Metrics: Accuracy, F1-score (macro average).
- Training for 100 epochs with evaluation on validation data after each epoch.

---

## Key Results
- Model trained to achieve high accuracy in detecting hate speech, despite dataset imbalance.
- Generated embeddings effectively represent contextual and positional information.

---

## Dependencies
- Python libraries: `pandas`, `numpy`, `torch`, `torchtext`, `nltk`, `spacy`, `sklearn`, `seaborn`, `matplotlib`.

---

## How to Use
1. Preprocess text using the provided tokenizer.
2. Load pre-trained embeddings and vocabulary.
3. Train the model using the `TextTransformer` class and validate performance.
4. Use the classifier for hate speech detection in real-world Twitter data.

---

## Future Improvements
- Experiment with advanced transformer architectures like GPT or T5.
- Incorporate additional datasets for better generalization.
- Address class imbalance using oversampling or synthetic data generation.
