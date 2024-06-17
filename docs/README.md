# AerialSeg-U-Net

![Python](https://img.shields.io/badge/Python-3.9.2-blue)
![Pillow](https://img.shields.io/badge/Pillow-8.3.2-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-yellow)
![numpy](https://img.shields.io/badge/numpy-1.21.2-blue)
![tensorflow](https://img.shields.io/badge/tensorflow-2.15.0-green)

This repository presents an implementation of a semantic segmentation model for aerial imagery using the U-Net architecture. The project aims to accurately segment various objects and features within aerial images, leveraging the powerful capabilities of the U-Net model. The model is trained to recognize and classify the following classes:

* **Building:**  #3C1098
* **Land (unpaved area):** #8429F6
* **Road:** #6EC1E4
* **Vegetation:** #FEDD3A
* **Water:** #E2A929
* **Unlabeled:** #9B9B9B

## Project Overview

This project implements a semantic segmentation model for aerial imagery using the U-Net architecture. The U-Net model is a convolutional neural network specifically designed for image segmentation tasks, achieving high accuracy in segmenting various objects and features within images. The code in this repository leverages TensorFlow to implement the U-Net model and perform segmentation on aerial imagery. 

### U-Net architecture

The U-Net architecture is characterized by its unique encoder-decoder structure. The encoder progressively downsamples the input image using convolutional and pooling layers, extracting hierarchical feature maps. The decoder then upsamples these features, gradually reconstructing the segmented image.  A key feature of U-Net is the use of skip connections, which combine feature maps from the encoder with corresponding decoder layers. This allows the model to retain fine-grained spatial information from the input, improving segmentation accuracy. 

<div style="text-align: center;">
  <img src="/docs/assets/unet.png" alt="New Image" title="New Image" style="width: 700px; max-width: 100%;" />
</div>

This architecture makes U-Net particularly effective for semantic segmentation tasks involving complex structures and boundaries, as seen in aerial imagery.  The U-Net architecture was originally proposed in the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 

### Dataset

This project utilizes the ["Semantic segmentation of aerial imagery dataset"](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery), hosted on `Kaggle`. This Public Domain dataset consists of aerial imagery of Dubai, UAE, captured by MBRSC satellites and annotated with pixel-wise semantic segmentation into 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles.

**Key Features of the Dataset:**

* **Content:** Satellite images of Dubai, UAE, segmented into six classes.
* **Classes:**
    * **Building:** #3C1098
    * **Land (unpaved area):** #8429F6
    * **Road:** #6EC1E4
    * **Vegetation:** #FEDD3A
    * **Water:** #E2A929
    * **Unlabeled:** #9B9B9B
* **Size:** 72 images grouped into 6 larger tiles.
* **Source:** The images were segmented by the trainees of the Roia Foundation in Syria.

This dataset is ideal for this project due to its:

* **Relevance:** The dataset provides real-world aerial images, suitable for training and evaluating a semantic segmentation model.
* **Complexity:** The diverse classes and varying scales of objects present a challenging but realistic segmentation task.
* **Availability:** The dataset is publicly available on Kaggle, making it easily accessible for research and development.


## Project Structure

**`Complete Notebook hosted on Kaggle:` [https://www.kaggle.com/code/eshansurendra/semantic-segmentation-using-u-net](https://www.kaggle.com/code/eshansurendra/semantic-segmentation-using-u-net)**

This notebook provides a comprehensive implementation of the semantic segmentation project using the U-Net model, covering data preprocessing, model building, training, and prediction steps. 

### Data Preparation

This section details the steps involved in preparing the dataset for training the U-Net model.  The dataset is first loaded, then preprocessed to ensure consistency and enhance training performance. Finally, the images and masks are divided into smaller patches to improve processing efficiency.

**1. Loading Images and Masks**

The code iterates through the dataset directory, reading images from subdirectories named 'images' and masks from subdirectories named 'masks'.  **It's crucial to ensure that the image and mask files are sorted in the same order to avoid mismatching pairs during training.** To achieve this, use the `sorted` method when listing the files:

```python
images = sorted(os.listdir(path))  # Sort images in the directory
masks = sorted(os.listdir(path))  # Sort masks in the directory
```

**2. Preprocessing**

* **Resizing:**  As images in the dataset have different sizes, they are cropped to the nearest size divisible by the patch size (256 in this case). This ensures that all images have dimensions that are multiples of the patch size, which is important for patching.
* **Normalization:** The image pixel values are normalized using a MinMaxScaler to scale them to the range [0, 1]. This helps to improve training stability and reduce the impact of large pixel values.

```python
image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest X size divisible by our patch size(256)
SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest Y size divisible by our patch size
image = Image.fromarray(image)
image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
image = np.array(image) 
```

**3. Patchification**

The `patchify` method is used to divide the preprocessed images and masks into smaller patches of size 256x256. This is crucial for efficient processing, as it allows the model to train on smaller chunks of data, which are easier to handle.  The code iterates through each patch, normalizing it (using the MinMaxScaler) and appending it to the image_dataset and mask_dataset lists.

```python

# patchify images
patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
for i in range(patches_img.shape[0]):
  for j in range(patches_img.shape[1]):
      
      single_patch_img = patches_img[i,j,:,:]
      
      #Use minmaxscaler instead of just dividing by 255. 
      single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
      
      #single_patch_img = (single_patch_img.astype('float32')) / 255. 
      single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
      image_dataset.append(single_patch_img)

```

After patchification, the image_dataset and mask_dataset contain a collection of patches ready for training. 

**Example Patch Dimensions:**

* **Tile 1:** 797 x 644 --> 768 x 512 --> 6 patches
* **Tile 2:** 509 x 544 --> 512 x 256 --> 2 patches
* **Tile 3:** 682 x 658 --> 512 x 512 --> 4 patches
* **Tile 4:** 1099 x 846 --> 1024 x 768 --> 12 patches
* **Tile 5:** 1126 x 1058 --> 1024 x 1024 --> 16 patches
* **Tile 6:** 859 x 838 --> 768 x 768 --> 9 patches
* **Tile 7:** 1817 x 2061 --> 1792 x 2048 --> 56 patches
* **Tile 8:** 2149 x 1479 --> 1280 x 2048 --> 40 patches
* **Total:** 1305 patches of size 256x256 (9 images per folder * 145 patches)

This process ensures that the dataset is structured in a way that optimizes training efficiency and model performance. 

### Model Building

This section focuses on the construction and configuration of the U-Net model, which will be used to perform semantic segmentation on aerial images.

**1. U-Net Architecture**

The U-Net architecture is implemented using TensorFlow's Keras API. The model consists of:

* **Encoder:** A series of convolutional and pooling layers to progressively extract features from the input image.
* **Decoder:** A series of upsampling and convolutional layers to reconstruct the segmented image using the extracted features.
* **Skip Connections:**  These connections link corresponding layers in the encoder and decoder, enabling the model to retain spatial information from the input image and improve segmentation accuracy.

**2. Model Compilation**

The U-Net model is compiled using the following parameters:

* **Optimizer:**  The `Adam optimizer` is selected for its ability to efficiently update model weights during training.
* **Loss Function:**  A custom loss function is defined to combine the **Dice loss** and **Focal loss**. 
    * **Dice Loss:** This loss function emphasizes the intersection over union (IoU) between the predicted and actual segmentation masks, making it suitable for segmenting images with imbalanced classes.
    * **Focal Loss:** This loss function helps to address the class imbalance problem by assigning higher weights to misclassified samples, improving model performance on minority classes.
* **Metrics:** The model is evaluated using two metrics:
    * **Accuracy:** A standard metric for classification tasks, measuring the percentage of correctly classified pixels.
    * **Jacard Coefficient (IoU):** A common metric for semantic segmentation, measuring the overlap between the predicted and actual segmentation masks.

```python
# Install segmentation_models: pip install segmentation_models
import segmentation_models as sm  

# Define custom loss function
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]  # Class weights for balanced training
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
```

```python
# Define Jacard coefficient metric
def jacard_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)
```
```python
# Compile the model
model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', jacard_coef])
```

**Training Preparation:**

* **One-Hot Encoding:** The target masks are converted from class labels to one-hot encoded vectors using `to_categorical` from Keras. This ensures that the model learns to predict probabilities for each class.
* **Train-Test Split:** The dataset is split into training and test sets using `train_test_split` to evaluate the model's performance on unseen data.

```python
n_classes = len(np.unique(labels))  # Number of unique classes
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)
```

This completes the model building process, preparing the model for training on the prepared dataset.

### Training

This section describes the process of training the U-Net model on the prepared dataset. 

**1. Training Loop**

The model is trained using the `fit` method from Keras. The training loop involves the following steps:

```python
history1 = model.fit(
    X_train, y_train, 
    batch_size = 16, 
    verbose=1, 
    epochs=50, 
    validation_data=(X_test, y_test), 
    shuffle=False
)
```

**2. Model Saving**

After training is complete, the trained model weights are saved to a file using the `save` method from Keras. This allows us to load and use the trained model for predictions later.

```python
model.save("/kaggle/working/satellite_standard_unet_100epochs.hdf5") 
```

### Prediction

This section focuses on using the trained U-Net model to make predictions on new images and visualize the results. 

**1. Model Loading**

The trained model is loaded from the saved file using the `load_model` function from Keras. It's important to specify the custom loss functions and metrics defined during training to ensure compatibility.

```python
model_path = "/kaggle/working/satellite_standard_unet_100epochs.hdf5"
custom_objects = {
    "dice_loss_plus_1focal_loss": total_loss,
    "jacard_coef": jacard_coef
}

# Load the model with custom objects
model = load_model(model_path, custom_objects=custom_objects)
```
**2. Prediction**

The loaded model is used to predict segmentation masks for new images. The `predict` method takes the input images as arguments and returns the predicted masks. 

```python
# Make predictions on the test set
y_pred = model.predict(X_test)
```

**3. Post-processing**

In this example, the predicted masks are processed by taking the argmax across the last dimension, effectively selecting the class with the highest probability at each pixel. This results in a single-channel mask where each pixel represents the predicted class.

```python
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)
```


## Results

The U-Net model, trained on the "Semantic segmentation of aerial imagery" dataset, achieved promising results on the segmentation task.  This section presents the key performance metrics and visualizations to illustrate the model's capabilities. 

<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="/docs/assets/leftgraph.png" alt="title-1" title="title-1" width="450" />
  <img src="/docs/assets/rightgraph.png" alt="title-2" title="title-2" width="450" />
</div>


**Performance Metrics:**

* **Test Accuracy:** 87%
* **Test Jaccard Coefficient (IoU):** 0.7308
* **Test Loss:** 0.8974
* **Validation Accuracy:** 84%
* **Validation Jaccard Coefficient (IoU):** 0.6869
* **Validation Loss:** 0.9149

These metrics indicate that the model demonstrates strong generalization ability, as it performs well on both the training and validation sets.  The high accuracy and IoU scores suggest that the model effectively learns to segment aerial images into distinct object categories.

**Visualization of Segmentation Results:**

Here are examples of segmentation predictions made by the model on test images:

![ex1](/docs/assets/ex1.png)

![ex2](/docs/assets/ex2.png)

![ex3](/docs/assets/ex3.png)

## Pretrained Models

Pretrained models are provided in the `pretrained_models` directory. You can load and use these models directly without training:

* `satellite_standard_unet_100epochs.keras`

## References

* **U-Net: Convolutional Networks for Biomedical Image Segmentation**
    * Olaf Ronneberger, Philipp Fischer, and Thomas Brox
    * [https://arxiv.org/pdf/1505.04597](https://arxiv.org/pdf/1505.04597)

* **Semantic segmentation of aerial imagery**
    * Satellite images of Dubai, the UAE segmented into 6 classes
    * [https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)
    **License:** CC0: Public Domain
    **Acknowledgements:** The images were segmented by the trainees of the Roia Foundation in Syria.

**For a more comprehensive understanding of the implementation, please refer to the complete notebook published on Kaggle:**

[https://www.kaggle.com/code/eshansurendra/semantic-segmentation-using-u-net/notebook#Semantic-segmentation-of-aerial-imagery-using-U-Net](https://www.kaggle.com/code/eshansurendra/semantic-segmentation-using-u-net/notebook#Semantic-segmentation-of-aerial-imagery-using-U-Net) 

## Contributing

Contributions are welcome! 

- **Bug Fixes:** If you find any bugs or issues, feel free to create an issue or submit a pull request.
- **Feature Enhancements:** If you have ideas for new features or improvements, don't hesitate to share them.

## License

This project is licensed under the [MIT License](LICENSE). 

[go to the top](#AerialSeg-U-Net)
