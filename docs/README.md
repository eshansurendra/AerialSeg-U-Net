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

## Project Structure

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
