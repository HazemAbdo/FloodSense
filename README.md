# 1. FloodSense:

Detecting post-flood damages in satellite imagery using machine learning and computer vision techniques

<p align="center">
<img src="/images/meme.jpg" width="500" height="300">
</p>

# 2. Project Pipeline:

![pipeline.png](/images/pipeline.png)

# 3. Classification task:

## 3.1 Data pre-processing:

### 3.1.1 Resizing:

we resize all the images to a size of 224*224 or 256*256 for these reasons:

1. Memory and computational efficiency
2. Reduce overfitting: Large images can contain a lot of irrelevant information that is not related to the classification task at hand.
3. Increase generalization
4. Not all pictures have the same size.

### 3.1.2 Normalization:

By dividing each pixel value by 255 for these reasons:

1. Improved model performance:  by reducing the impact of differences in the scale or range of input features.
2. Preventing numerical instability
3. Facilitating feature interpretation: Normalization can make it easier to interpret the importance of different input features.

### 3.1.3 Augmentation:

Since data is limited augmentations are applied to enhance the diversity of training data.

1. Flips the image horizontally (left to right) randomly.
2. Flips the image vertically (upside down) randomly.
3. Rotates the image by a random angle within the specified range.
4. Randomly altering the color attributes of the image.
5. Standardizing the input data to have zero mean and unit variance (done in validation & test as well).

## 3.2 Feature extraction and selection:

### 3.2.1 Machine learning approach:

1. Histogram of Oriented Gradients (HOG) :
   - which works by dividing an image into small cells and computing the gradient (direction and magnitude of change) of pixel intensities within each cell.
2. Gray Level Co-occurrence Matrix (GLCM):
   - which works by computing the co-occurrence frequency of pairs of pixel intensity values in an image.
3. Local Binary Patterns (LBP):
   - which works by comparing the intensity of each pixel in an image to the intensity of its neighboring pixels and then encoding the result as a binary pattern.
4. Oriented FAST and Rotated BRIEF (ORB):
   - ORB detects key points in an image using the FAST algorithm and then computes binary feature descriptors using the BRIEF algorithm.
5. Color-based features:
   - Two common types of color-based features are mean and standard deviation, and color histograms.
     - Mean and standard deviation features capture information about the distribution of pixel values in an image.
     - Color histograms, on the other hand, capture information about the frequency of pixel values in an image.
6. Principle component analysis (PCA):
   - To overcome the "curse of dimensionality".
   - To reduce the complexity of the model and reduce the risk of overfitting.

### 3.2 .2 Deep learning approach:

1. **ResNet** uses residual blocks and skips connections to extract features from input images. This approach allows the network to learn residual functions that can approximate the underlying mapping between the input and output

2. **EfficientNet** is a family of convolutional neural network architectures designed for high performance and computational efficiency. which utilizes a compound scaling method to balance depth, width, and resolution achieving optimal trade-offs.

3. **MobileNet** is a lightweight convolutional neural network architecture designed for mobile and embedded devices, which uses depth-wise separable convolutions to reduce the computational complexity while maintaining good accuracy and suitable choice when dealing with limited data due to its ability to generalize well even with smaller training datasets since depth-wise separable convolutions reduce the number of parameters and improve the model's ability to learn from limited samples.

## 3.3 Model training:

### 3.3.1 Machine learning approach:

1. Random forest classifier :
   - `max_depth=5`: Sets the maximum depth of each decision tree in the ensemble model.
   - `n_estimators=200`: Sets the number of decision trees in the ensemble model.
   - `min_samples_leaf=5`: Sets the minimum number of samples required to be at a leaf node in each decision tree.
2. SVM
   - `C=0.001`: Sets the regularization strength for the SVM model. A smaller value of C results in stronger regularization and a simpler model.
   - `kernel='linear'`: Sets the type of kernel function used in the SVM model. In this case, a linear kernel is used, which is appropriate for linearly separable datasets.
3. 2 layers neural network
   - `regularizers_l2=0.01`: sets the strength of `L2` regularization penalty in the neural network.
   - `dropout=0.7`: Sets the probability of randomly dropping out neurons in the network during training to prevent overfitting.
   - `early_stopping_monitor='val_loss'`: Monitors the validation loss during training for early stopping.
   - `patience_val=3`: Sets the number of epochs to wait before stopping the training process if the model's performance on the validation set does not improve.
   - `optimizer_name='adam'`: Sets the name of the optimizer algorithm used to update the weights of the neural network during training.
   - `mat=metrics.AUC()`: Sets the metric used to evaluate the performance of the neural network during training.
   - `epochs_num=10`: Sets the number of epochs (passes over the training dataset) during training.
   - `batch_size=32`: Sets the number of training examples used in one iteration of the training process.

### 3.3.2 Deep learning approach:

1. ResNet50
2. ResNet18

- we use both models in the pre-trained flavor and not pre-trained with a better performance in the pre-trained one as pre-trained models are often better at achieving high performance because they have already learned to recognize general patterns and features from a large and diverse dataset
- here are our choice of the loss function and optimizer which are based on this [paper](https://arxiv.org/pdf/2105.08655.pdf)
  ```
  loss_fn = BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
  ```
- n_epochs=100,early_stopping_tolerance=5,early_stopping_threshold=0.03

1. ResNet14t

- ResNet18 and 50 showed bad generalization to the validation however ResNet14t with less number of layers showed better performance and generalization.
- we use the model in the pretrained flavor and without freezing any layers.

```python
loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
```

1. EfficientNet_b0

- We tried this model due to it generalization capability since we have limited dataset.
- We tried different versions of EfficientNet b0, b1, b2, b3 and b4 and b0 shows the best performance and generalization due to it less number of parameters.
- we use the model in the pretrained flavor and without freezing any layers.

```
loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
```

1. MobileNetV3_large_100

- We tried this model because it was designed for mobiles and embedded systems and has a generalization capability since it tries to reduce the number of parameters while achieving good accuracy.
- We tried different versions of MobileNet V2 and V3 however MobileNetV3_large_100 shows the best performance and generalization.
- we use the model in the pre-trained flavor without freezing any layers.

```
loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
```

1. Ensemble

- We Ensembled our three best models ResNet14t, EfficientNet_b0 and MobileNetV3_large_100.
- The main goal here is to decrease the variance and be more robust to outliers so we used average ensemble.

## 3.4 Evaluation:

### 3.4.1 Machine learning approaches without using features:

- 2 layer NN

  ![Untitled](/images/1.png)

- Random forest classifier

  ![Untitled](/images/2.png)

- SVM
  ![Untitled](/images/3.png)

### 3.4.2 Machine learning approach using features:

- 2 layer NN
  ![Untitled](/images/4.png)
- Random forest classifier

![Untitled](/images/5.png)

- SVM

![Untitled](/images/6.png)

### 3.4.3 Deep learning approach:

1. ResNet50
2. ResNet18

![Untitled](/images/7.png)

1. ResNet14t

|                 | Predicted Negative | Predicted Positive |
| --------------- | ------------------ | ------------------ |
| Actual Negative | 93                 | 0                  |
| Actual Positive | 1                  | 91                 |

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0.0          | 0.99      | 1.00   | 0.99     | 93      |
| 1.0          | 1.00      | 0.99   | 0.99     | 92      |
| Accuracy     |           |        | 0.99     | 185     |
| Macro avg    | 0.99      | 0.99   | 0.99     | 185     |
| Weighted avg | 0.99      | 0.99   | 0.99     | 185     |

|                        | Error Rates |
| ---------------------- | ----------- |
| Omission error rate:   | 0.01        |
| Commission error rate: | 0.0         |

1. EffiecientNet_b0

|                 | Predicted Negative | Predicted Positive |
| --------------- | ------------------ | ------------------ |
| Actual Negative | 93                 | 0                  |
| Actual Positive | 0                  | 92                 |

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0.0          | 1.00      | 1.00   | 1.00     | 93      |
| 1.0          | 1.00      | 1.00   | 1.00     | 92      |
| Accuracy     |           |        | 1.00     | 185     |
| Macro avg    | 1.00      | 1.00   | 1.00     | 185     |
| Weighted avg | 1.00      | 1.00   | 1.00     | 185     |

|                        | Error Rates |
| ---------------------- | ----------- |
| Omission error rate:   | 0.0         |
| Commission error rate: | 0.0         |

1. MobileNetV3_large_100

|                 | Predicted Negative | Predicted Positive |
| --------------- | ------------------ | ------------------ |
| Actual Negative | 93                 | 0                  |
| Actual Positive | 1                  | 91                 |

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0.0          | 0.99      | 1.00   | 0.99     | 93      |
| 1.0          | 1.00      | 0.99   | 0.99     | 92      |
| Accuracy     |           |        | 0.99     | 185     |
| Macro avg    | 0.99      | 0.99   | 0.99     | 185     |
| Weighted avg | 0.99      | 0.99   | 0.99     | 185     |

|                        | Error Rates |
| ---------------------- | ----------- |
| Omission error rate:   | 0.01        |
| Commission error rate: | 0.0         |

1. Ensemble

|                 | Predicted Negative | Predicted Positive |
| --------------- | ------------------ | ------------------ |
| Actual Negative | 93                 | 0                  |
| Actual Positive | 0                  | 92                 |

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0.0          | 1.00      | 1.00   | 1.00     | 93      |
| 1.0          | 1.00      | 1.00   | 1.00     | 92      |
| Accuracy     |           |        | 1.00     | 185     |
| Macro avg    | 1.00      | 1.00   | 1.00     | 185     |
| Weighted avg | 1.00      | 1.00   | 1.00     | 185     |

|                        | Error Rates |
| ---------------------- | ----------- |
| Omission error rate:   | 0.0         |
| Commission error rate: | 0.0         |

# 4. Segmentation:

## 4.1 **Data preprocessing**

1. Dilation: The image is dilated using a 3x3 kernel to connect gaps in the objects.
2. Erosion: The image is eroded using a 3x3 kernel to shrink the image back.
3. Blur: Gaussian blur is applied to the image using a 3x3 kernel to reduce noise.
4. CLAHE: a contrast enhancement technique that improves the contrast of an image.
5. Conversion: return the image in different formats RGB, HSV, and LAB.

## 4.2 K-means + calinski harabasz score

1. The Calinski-Harabasz score assesses how well the data points are grouped into clusters based on their similarity and separation.
   1. Much faster than Silhouette's score which is too slow.
2. Apply k-means multiple times using different numbers of clusters and choose the number of clusters that achieves the best calinski harabasz score.
3. Apply k-means multiple times using different formats of the image RGB, HSV, and LAB then choose the format that achieves the best calinski harabasz score.

# 5. The final Model:

The model used for submission is the ensemble model that averages ResNet14t, EfficientNet_b0, and MobileNetV3_large_100.

Weights:

[https://drive.google.com/drive/folders/1uETvV6byEbIl_uZwLUVfrRe62488wTJU?usp=share_link](https://drive.google.com/drive/folders/1uETvV6byEbIl_uZwLUVfrRe62488wTJU?usp=share_link)
