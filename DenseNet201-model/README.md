# Skin Cancer Classification with DenseNet201

## ISIC Dataset

### Dataset Link: [Skin cancer dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

This set consists of 2357 images of malignant and benign oncological diseases, which were formed from The International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following diseases:

- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis
- seborrheic keratosis
- squamous cell carcinoma
- vascular lesion

## Model 1: DenseNet201 without Fine-Tuning – Adam optimizer

This model uses a pre-trained DenseNet201 as a fixed feature extractor. The convolutional base is not trained; only the weights of the newly added classifier head are updated.

### Architecture

> Base Model: DenseNet201 (pre-trained on ImageNet, include_top=False). The weights of this base are frozen.
> Classifier Head:
> Flatten()
> BatchNormalization()
> Dropout(0.5)
> Dense(512, activation='relu')
> Dropout(0.3)
> Dense(9, activation='softmax')

### Training Parameters

Optimizer: Adam with a learning rate of 0.001
Epochs: 50 (with EarlyStopping patience of 5)
Batch Size: 32
Callbacks: ReduceLROnPlateau to decrease the learning rate if validation loss plateaus, and EarlyStopping to prevent overfitting.

### Performance

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 73.23% |
| Training Loss     | 0.6960 |
| Test Accuracy     | 71.33% |
| Test Loss         | 0.7670 |

### Classification Report Analysis

This classification report indicates that the model achieves an overall “accuracy of 71%”, which is moderate but shows room for improvement. Some classes, such as “vascular lesion”, “seborrheic keratosis”, and “dermatofibroma”, perform well with high precision, recall, and F1-scores, indicating that the model identifies these lesions reliably. In contrast, “melanoma” shows very low recall (0.15) and F1-score (0.23), meaning the model fails to detect the majority of melanoma cases—a critical issue in medical diagnosis. Other classes like “pigmented benign keratosis”, “actinic keratosis”, and “nevus” have moderate performance, with recall slightly higher than precision, suggesting some over-prediction of these classes. Overall, while the model captures common lesion patterns effectively, it struggles with rare or visually similar classes, highlighting the need for more targeted data augmentation, class balancing, or fine-tuning to improve detection of high-risk categories like melanoma.

### Model 1 Summary:

The model achieves a test accuracy of 71.36%, indicating moderate overall performance in classifying skin lesion images. Its precision of 70.26% shows that most of the predicted positive cases are correct, while the recall of 71.41% suggests it is detecting a reasonable portion of actual positive cases. The F1-score of 69.33% reflects a balanced trade-off between precision and recall, though there is room for improvement in consistency across classes. The Cohen’s Kappa score of 0.6777 indicates substantial agreement between the model’s predictions and the ground truth, confirming that the model performs significantly better than random chance. Overall, while the model shows decent predictive capability, further fine-tuning or architectural optimization may be required for more reliable medical use.

## Model 2: DenseNet201 with Fine-Tuning – Adam optimizer

This model employs a two-phase training strategy: first, it trains only the classifier head (transfer learning), and then it unfreezes and fine-tunes the top layers of the DenseNet201 base.

## Architecture

> Base Model: DenseNet201 (pre-trained on ImageNet, include_top=False)
> Classifier Head:
> Flatten()
> BatchNormalization()
> Dropout(0.5)
> Dense(512, activation='relu')
> Dropout(0.6)
> Dense(9, activation='softmax')

### Training Parameters

#### Phase 1:

Base model frozen
Optimizer: Adam (lr = 0.001)
Epochs: 15
Batch Size: 16

#### Phase 2:

Unfreeze top 10 layers
Optimizer: Adam (lr = 0.0001)
Epochs: 10
Batch Size: 16

### Performance

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 92.05% |
| Training Loss     | 0.2453 |
| Test Accuracy     | 80.27% |
| Test Loss         | 0.5824 |

### Classification Report Analysis

This classification report shows that your model achieves strong and balanced performance across most of the nine skin lesion classes, with an overall accuracy of 80%. Classes like vascular lesion, dermatofibroma, seborrheic keratosis, and basal cell carcinoma perform exceptionally well, with F1-scores between 0.81 and 0.99, meaning the model identifies these lesions very reliably. However, a few more challenging classes show lower recall, especially melanoma (0.57 recall) and pigmented benign keratosis (0.71 recall), indicating the model misses a noticeable number of true cases in these categories. This is common in skin lesion classification because these classes visually overlap with others. Classes like actinic keratosis and squamous cell carcinoma show good recall but slightly lower precision, meaning the model sometimes overpredicts them. The macro and weighted averages (0.80) confirm consistent performance without major class imbalance issues. Overall, the model is strong but could still benefit from more targeted augmentation or fine-tuning specifically for melanoma and visually similar lesion classes.

### Model 2 Summary:

The model is performing quite well, with all major metrics around 80%, indicating that it is learning meaningful patterns and generalizing reasonably well. The performance is balanced — meaning the model is not heavily biased toward any particular class.

## Model 3: DenseNet201 with SGD Optimizer

This model explores the use of the SGD optimizer instead of Adam, with a simpler architecture for the classifier head.

### Architecture

> Base Model: DenseNet201 (pre-trained on ImageNet, include_top=False)
> Classifier Head:
> Flatten()
> Dropout(0.6)
> Dense(512, activation='relu')
> Dense(9, activation='softmax')

## Training Parameters

Optimizer: SGD (lr = 0.001, momentum = 0.9)
Epochs: 20
Batch Size: 32
Callbacks: ReduceLROnPlateau( patience=3, factor=0.5, verbose=1, min_lr=0.00001)

### Performance

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 99.38% |
| Training Loss     | 0.0131 |
| Test Accuracy     | 90.22% |
| Test Loss         | 0.4766 |

### Classification Report Analysis

The classification report shows that the model performs strongly across all nine skin-lesion categories, achieving an overall accuracy of 90% on the test set. Most classes, such as pigmented benign keratosis, vascular lesion, squamous cell carcinoma, basal cell carcinoma, dermatofibroma, and actinic keratosis, show high precision, recall, and F1-scores, indicating reliable and consistent predictions. However, classes like melanoma, seborrheic keratosis, and nevus have slightly lower recall values, meaning the model misses some samples from these categories. Despite this, the macro and weighted averages remain around 0.90, confirming that the model maintains balanced and stable performance across both common and less frequent classes.

### Model 3 Summary:

The model demonstrates strong overall performance, achieving an accuracy of 90.22%, which indicates that most predictions are correct. Its precision of 90.53% shows that when the model predicts a class, it is usually correct, while the recall of 90.60% confirms that it successfully identifies most of the actual positive cases. The F1-score of 90.34% reflects a good balance between precision and recall, making the model reliable across different classes. Additionally, the Kappa score of 0.89 indicates excellent agreement between the model’s predictions and the true labels, showing that the predictions are far better than random chance.

## Model 4: DenseNet201 with Global Average Pooling

This model replaces the Flatten layer with a GlobalAveragePooling2D layer, which is known to reduce overfitting. It also uses a two-phase training approach similar to Model 2.

### Architecture

> Base Model: DenseNet201 (pre-trained on ImageNet, include_top=False)
> Classifier Head:
> GlobalAveragePooling2D()
> BatchNormalization()
> Dropout(0.5)
> Dense(256, activation='relu')
> Dropout(0.6)
> Dense(9, activation='softmax')

### Training Parameters

#### Phase 1:

Base frozen
Optimizer: Adam (lr = 1e-3)
Epochs: 15
Batch Size: 16

#### Phase 2: Fine Tuning

Unfreeze last 20 layers
Optimizer: Adam (lr = 1e-5)
Epochs: 10
Batch Size: 16

### Performance

| Metric            | Value  |
| ----------------- | ------ |
| Training Accuracy | 83.57% |
| Training Loss     | 0.5022 |
| Test Accuracy     | 76.02% |
| Test Loss         | 0.6754 |

### Classification Report Analysis

This classification report shows that the model achieves an overall accuracy of 76%, indicating moderate performance across the nine skin lesion classes. Some classes, like vascular lesion and dermatofibroma, perform exceptionally well with F1-scores around 0.91–0.98, showing the model can reliably identify these lesions. Classes such as melanoma, pigmented benign keratosis, and nevus have lower recall (0.45–0.67), meaning the model misses a significant number of true cases, although precision is somewhat higher, suggesting some over-prediction for these classes. Other classes like actinic keratosis, seborrheic keratosis, and basal cell carcinoma show decent balance between precision and recall. Overall, the model is effective for visually distinct lesions but struggles with clinically similar or rarer classes, highlighting the need for further fine-tuning, augmentation, or class-specific strategies to improve detection of high-risk lesions like melanoma.

### Model 4 Summary:

The model achieves an accuracy of 76.02%, showing solid performance in correctly classifying skin lesion images. With a precision of 76.35%, the model reliably identifies positive cases with relatively few false positives, while a recall of 75.92% indicates that it captures most true cases effectively. The F1-score of 75.50% reflects a good balance between precision and recall, demonstrating consistent behavior across different classes. Additionally, the Cohen’s Kappa score of 0.7301 signifies substantial agreement with the ground truth, confirming that the model performs well beyond random chance. Overall, this model shows strong and dependable classification capability, with improved reliability compared to lower‐accuracy variants.

## Densenet201 Models Summary:

| Model                                                  | Accuracy | Precision | Recall | F1-score | Kappa Score |
| ------------------------------------------------------ | -------- | --------- | ------ | -------- | ----------- |
| Densenet201 – Adam optimizer                           | 71.36    | 0.7026    | 0.7141 | 0.6933   | 0.6777      |
| Densenet201 – Adam optimizer with fine tuning          | 80.27    | 0.8083    | 0.8057 | 0.8019   | 0.7781      |
| Densenet201 – SGD optimizer                            | 90.22    | 0.9053    | 0.9060 | 0.9034   | 0.8900      |
| Densenet201 – Fine-tuning with GlobalMaxPooling + Adam | 76.02    | 0.7635    | 0.7592 | 0.7550   | 0.7301      |
