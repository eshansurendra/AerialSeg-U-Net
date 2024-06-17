# AerialSeg-U-Net

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
