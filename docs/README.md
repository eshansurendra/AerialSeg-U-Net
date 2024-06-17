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

![U-Net architecture](/docs/assets/unet.png)

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

The code iterates through the dataset directory, reading images from subdirectories named 'images' and masks from subdirectories named 'masks'. 

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
