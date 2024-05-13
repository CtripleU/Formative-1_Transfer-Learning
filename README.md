# Applying Transfer Learning to Glaucoma Detection

This project explores the application of transfer learning for the detection of glaucoma.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- 


## Repository Structure:
transfer_learning_glaucoma/
│
├── README.md
│
├── Transfer_Learning_Glaucoma.ipynb


## Problem Statement and Dataset

**Problem Statement:**  The objective is to develop an accurate image-based classifier that can distinguish between images of eyes with glaucoma and healthy eyes. Early detection of glaucoma is crucial for preventing vision loss, making this a valuable tool in ophthalmology.

**Dataset:** The data used for this task is the "partitioned_by_hospital" segment of the [RIM-ONE DL] (https://www.ias-iss.org/ojs/IAS/article/view/2346) dataset. It consists of retinal scans categorized into "glaucoma" and "normal" classes. The data is structured into two folders:

* **train_data:**  Contains images for training the models. 
    * `/train_data/glaucoma`
    * `/train_data/normal`
* **test_data:** Contains images for evaluating the trained models.
    * `/test_data/glaucoma`
    * `/test_data/normal`

## Pre-trained Models and Justification

Three pre-trained models, known for their strong performance in image classification tasks, have been chosen:

1. **VGG16:**
    * **Architecture:** VGG16 is known for its simple architecture using a stack of convolutional layers with small 3x3 filters. This simplicity makes it relatively easy to understand and implement. 
    * **Suitability for Glaucoma Detection:** While trained on natural images, VGG16's ability to extract general image features makes it a good starting point for fine-tuning on medical images. The learned features, such as edges, textures, and shapes, are likely to be relevant for glaucoma detection as well. 

2. **ResNet50:**
    * **Architecture:** ResNet50 utilizes residual connections (skip connections) to address the vanishing gradient problem, enabling the training of much deeper networks than VGG16. 
    * **Suitability for Glaucoma Detection:** Its superior performance and ability to extract more intricate features make it highly relevant for fine-tuning on specialized datasets like glaucoma images. The deeper architecture potentially allows for the detection of subtle patterns in the eye that might be missed by simpler models.

3. **InceptionV3:**
    * **Architecture:** InceptionV3 incorporates inception modules, which utilize multiple filter sizes to capture features at different scales within an image. This allows the model to analyze the image from different perspectives.
    * **Suitability for Glaucoma Detection:** The use of multiple filter sizes within inception modules could be particularly beneficial in detecting subtle patterns and variations in eye images that are indicative of glaucoma.  

## Fine-tuning Process and Rationale

1. **Loading Pre-trained Models:** The selected models (VGG16, ResNet50, InceptionV3) are loaded with weights pre-trained on the ImageNet dataset. This provides a strong foundation of general image features.

2. **Freezing Layers:** The initial convolutional layers of each model are frozen to preserve the learned features from ImageNet. This prevents the disruption of these general features while training on a smaller, specialized dataset. However, the last few convolutional layers are unfrozen to allow the model to adjust to the specifics of glaucoma images. The number of layers to unfreeze is a hyperparameter that can be tuned for optimal performance.

3. **Adding New Layers:** New fully connected layers are added on top of the convolutional base. These layers will learn specific features relevant to glaucoma detection from the dataset. The added layers consist of:
    * A flattening layer: Converts the multi-dimensional output of the convolutional base into a one-dimensional vector.
    * Dense layers: One or more dense layers with ReLU activation functions are used to learn complex relationships between the extracted features and glaucoma classification.
    * Output layer: A final dense layer with a sigmoid activation function outputs the probability of the image belonging to the "glaucoma" class.

4. **Training the Modified Models:** The models are trained using the Adam optimizer, which is known for its efficiency and effectiveness in training deep learning models. The Adam optimizer adapts the learning rate throughout the training process, leading to faster convergence. Early stopping is employed to prevent overfitting, halting the training process if the validation loss stops improving for a set number of epochs.

## Evaluation Metrics

The performance of the fine-tuned models is assessed using the following metrics:

* **Accuracy:** The overall percentage of correctly classified images.
* **Loss:** A measure of the model's error during training, quantifying how well the model's predictions match the true labels. It's calculated using the binary cross-entropy loss function. 
* **Precision:** Out of all the images predicted as "glaucoma," what percentage was actually glaucoma? 
* **Recall:** Out of all the actual "glaucoma" images, what percentage did the model correctly identify?
* **F1 score:**  The harmonic mean of precision and recall, providing a balanced measure of the model's performance, especially in cases of class imbalance.

## Results 

The performance of the three fine-tuned models on the glaucoma detection task is summarized in the following table:

| Model    | Accuracy | Loss | Precision | Recall | F1 score |
| -------- | -------- | -------- | -------- | -------- | -------- |
| VGG16      | 0.6609  | 0.7347 | 0.6766  | 0.9576 | 0.7930  |
| ResNet50     | 0.6782  | 0.6837 | 0.6782  | 1.0000 | 0.8.82  |
| InceptionV3 | 0.4425 | 2.4639 | 0.6235  | 0.4492 | 0.5222  |


## Discussion of Findings

The results presented in the table reveal that the fine-tuned models achieved relatively low performance on the glaucoma detection task. While ResNet50 slightly outperformed VGG16 and InceptionV3, none of the models achieved a satisfactory level of accuracy. 

* **Comparison of Model Performance:**  ResNet50 achieved the highest accuracy (67.82%) and F1 score (0.8082), suggesting it was slightly better at classifying glaucoma. However, all three models exhibited accuracies barely above random chance, indicating a significant challenge in discerning the features associated with glaucoma. 

* **Impact of Preprocessing and Fine-tuning:**  While preprocessing and fine-tuning were implemented, their impact on improving performance appears limited based on these results. This suggests that the features extracted from the pre-trained models and the chosen preprocessing steps were not sufficient to capture the subtle visual cues crucial for glaucoma identification. 

* **Strengths and Limitations of Transfer Learning:**  In this context, the limitations of transfer learning are more evident. The pre-trained models, despite their success on ImageNet, might not have learned features that generalize well to the specialized domain of ophthalmological images. The relatively low accuracy suggests a considerable domain mismatch between natural images and glaucoma images.

* **Potential Reasons for Underperformance:**
    * **Dataset Size and Quality:** The dataset may be too small or lack diversity, hindering the models' ability to learn effectively. The quality of images (resolution, lighting, presence of artifacts) could also impact performance.
    * **Choice of Preprocessing Techniques:** The selected preprocessing steps might not be optimal for highlighting glaucoma-related features. More specialized preprocessing, in consultation with ophthalmologists, could be beneficial.
    * **Limited Fine-tuning:**  Fine-tuning only the last few layers of each model might not be sufficient. Experimenting with unfreezing more layers or adopting a different fine-tuning strategy could be explored.
    * **Class Imbalance:** If the dataset has a significant imbalance between glaucoma and normal images, it can bias the model and lead to poor performance on the minority class. Addressing this imbalance through data augmentation or specific loss functions could help.

* **Future Directions:** 
    * **Larger and More Diverse Dataset:**  A larger dataset with a greater variety of glaucoma cases and healthy examples would likely improve the models' ability to learn. 
    * **Specialized Preprocessing:** Collaborating with ophthalmologists to develop more targeted preprocessing techniques that enhance glaucoma-specific features.
    * **Alternative Architectures:** Exploring other pre-trained models or custom architectures specifically designed for medical image analysis.
    * **Hyperparameter Tuning:** Systematically exploring different learning rates, batch sizes, and the number of unfrozen layers during fine-tuning to optimize performance.

The relatively poor performance highlights the complexities of applying transfer learning to specialized medical imaging tasks.  Further research and experimentation are necessary to develop more accurate and reliable glaucoma detection models using deep learning. 


## Conclusion

This project aimed to leverage transfer learning for the development of a glaucoma detection model. However, the results indicate that directly applying pre-trained models like VGG16, ResNet50, and InceptionV3, even with fine-tuning, did not achieve satisfactory accuracy for this task. The relatively low performance highlights the challenges of transferring knowledge from natural image datasets to the specialized domain of ophthalmological images. 

The findings suggest that further research is needed to develop more effective glaucoma detection models. This could involve exploring alternative pre-trained models better suited for medical imaging, designing custom architectures, and implementing more specialized preprocessing techniques in consultation with ophthalmologists. Additionally, a larger and more diverse dataset would likely be beneficial to improve the models' ability to learn relevant features. 

While this project did not achieve high accuracy, it served as a valuable learning experience, demonstrating the importance of careful consideration of dataset characteristics, domain-specific preprocessing, and the potential limitations of transfer learning when applied to specialized medical imaging problems.



