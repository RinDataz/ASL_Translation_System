# Yad-Tech: Real-Time ASL Recognition System

![image](https://github.com/user-attachments/assets/a291aaf8-d2fe-454e-9f3e-b598a16eeb43)

![image](https://github.com/user-attachments/assets/583b4ad6-5f54-492b-a6d6-af7771049c26)


## Overview
Welcome to the Yad-Tech project repository! This project was developed as part of the **Samsung x Misk Innovation Campus** program. Our aim is to bridge the communication gap between the Deaf and hearing communities by translating American Sign Language (ASL) gestures into text in real-time. This tool also serves as an educational resource for learning ASL.

Yad-Tech combines Convolutional Neural Networks (CNNs) for image-based gesture recognition and a Random Forest model for real-time prediction adjustments. By integrating these models with **Mediapipe** for hand landmark detection and **OpenCV** for webcam support, we created a system that is both highly accurate and efficient in live settings.

## Data Analysis
Our initial focus was on the ASL alphabet, including 29 classes: each letter and common functional signs like "space," "delete," and "nothing." The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/code?datasetId=23079&searchQuery=cnn), includes over 87,000 images, balanced across classes. To address class imbalance and enhance recognition, we applied data augmentation for underrepresented classes. 


![image](https://github.com/user-attachments/assets/923229fe-b951-4db7-9a92-c01e658c6a7e)

![image](https://github.com/user-attachments/assets/36407e6f-4a7d-4275-81cf-868f6924ba0a)

![image](https://github.com/user-attachments/assets/8b9bc1ad-9a16-4bfa-96ba-90e6300487af)


## Modeling

### [CNN Model](https://github.com/RinDataz/ASL_Translation_System/tree/main/CNN_Model)
The CNN model was built using a Keras Sequential architecture for high accuracy in image classification. After preprocessing and splitting the dataset into training and test sets, the model achieved remarkable results:
- **Training Accuracy**: 99.6%
- **Test Accuracy**: 98%

This high performance indicates the CNN’s robustness in identifying static ASL gestures accurately.


![image](https://github.com/user-attachments/assets/aeb06877-fbac-47db-b3e9-c320af9a928d)

![image](https://github.com/user-attachments/assets/16a6674e-e6f0-473d-a173-80d854cc3775)

![image](https://github.com/user-attachments/assets/a3e20589-a185-4525-abdd-668f79363cba)

The model achieved excellent performance across all ASL alphabet classes, with an overall accuracy of 98%. Key metrics, including precision, recall, and F1-score, are consistently high, with most classes scoring above 0.97 in each metric.

The **confusion matrix** shows that the model performs well, with high accuracy across most ASL alphabet classes, evidenced by strong values along the diagonal. Minor misclassifications occur between some visually similar classes, but overall, the model generalizes effectively with minimal errors. Fine-tuning or targeted data augmentation could further improve accuracy for the few challenging classes.


**Model Accuracy**

The training accuracy rapidly increased, reaching nearly 100% within the first few epochs. This indicates that the model quickly learned patterns in the training data.
Validation accuracy shows an overall upward trend, although it fluctuates slightly, especially in the early epochs. By the final epochs, validation accuracy stabilizes and aligns closely with training accuracy, suggesting that the model generalizes well without overfitting.


**Model Loss**

The training loss consistently decreased and approached zero, indicating that the model is minimizing errors on the training data effectively.
The validation loss shows a steep decline initially, mirroring the validation accuracy improvement. While it fluctuates slightly across epochs, it ultimately stabilizes close to the training loss, which further indicates good generalization.


### [Random Forest Model](https://github.com/RinDataz/ASL_Translation_System/tree/main/RandomForestModel)
To enhance real-time performance, we used a Random Forest Classifier trained on hand landmark data extracted using Mediapipe. This model operates on a confidence threshold to ensure only high-certainty predictions are displayed. The Random Forest model achieved:
- **Accuracy**: 99.9% on validation data

![image](https://github.com/user-attachments/assets/fddfe1e8-a1ed-4aaa-b5e2-7783f38ad0b5)

Accuracy: The model achieved an accuracy of 99.8%, which indicates it’s highly effective at recognizing and classifying signs.
Precision, Recall, and F1-Score: Each class shows nearly perfect scores (close to 1.00) in precision, recall, and F1-score, confirming consistent performance across all gestures. This balanced performance is crucial for real-time applications to avoid biases toward any specific gesture.

![image](https://github.com/user-attachments/assets/43131a54-1c7b-43f2-bb0f-1721af723275)

The **confusion matrix** shows that the model performs exceptionally well across all classes, with almost perfect classification for each sign. The model misclassifies very few instances, as shown by the small number of off-diagonal values, suggesting minimal errors.


The integration of CNN and Random Forest allows the system to operate smoothly in real-time, effectively balancing accuracy and efficiency.

## [User Interface](https://github.com/RinDataz/ASL_Translation_System/tree/main/RandomForestModel/Web_testing)

<img width="948" alt="Screenshot 2024-11-12 113338" src="https://github.com/user-attachments/assets/75218da9-b9c2-4645-8549-329b8f7a166a">


For real-time ASL gesture recognition, we leveraged a Random Forest model combined with a webcam feed and the Mediapipe library for hand tracking. Mediapipe extracts hand landmarks, allowing us to capture and preprocess real-time gesture data from users’ hands, which is then passed through the Random Forest model for prediction. The model classifies each gesture and updates the output text based on the identified sign language letter or action.
Key steps in the real-time recognition process:


•	[Hand Landmark Detection](https://github.com/RinDataz/ASL_Translation_System/blob/main/landmark_data.pkl): The Mediapipe library captures and tracks hand landmarks in each video frame, generating x, y coordinates of key points on the user's hand.

•	Data Preparation: To ensure consistency with our model’s input requirements, the landmark data is normalized and arranged to match the model’s expected input length.

•	Prediction and Text Update: The preprocessed data is passed to the trained Random Forest model, which predicts the ASL letter or gesture. When the confidence level exceeds a specified threshold (e.g., 50%), the system updates the text output accordingly. For actions like "space" and "delete," the text is modified with appropriate spacing or character removal.

## Testing and Improvements

The Yad-tech ASL recognition system was rigorously tested on both pre-split test data and additional user-generated images to ensure robust performance. The CNN model achieved a high accuracy of 98%, with misclassifications primarily in visually similar gestures. The Random Forest model, paired with MediaPipe for real-time recognition, showed reliable accuracy during live webcam testing.


## Key Improvements:

•	Hyperparameter Tuning and Dropout Layers: Adjustments to CNN hyperparameters and the addition of dropout layers helped control overfitting, boosting generalization.

•	Confidence Threshold Optimization: Fine-tuning confidence thresholds in the Random Forest model reduced noise, ensuring reliable character insertion in real-time.

•	User Feedback: User testing informed refinements to delay settings, making text insertion smoother for real-time interaction.


## Conclusions
Yad-Tech combines advanced image classification and real-time processing to create a powerful tool for ASL translation. Our CNN and Random Forest models achieved high accuracy rates, with the Sequential model reaching a test accuracy of 98% and the Random Forest delivering near-perfect real-time accuracy. These models have the potential to enhance communication and learning, bridging a crucial gap between the Deaf and hearing communities.

### Future Improvements
To further improve robustness, we aim to:
- Expand the dataset to include more diverse gestures and variations.
- Extend the system to recognize dynamic gestures and incorporate regional sign languages.

## About
An ASL Translator using CNN for image classification and real-time hand detection with OpenCV and MediaPipe. Achieves high accuracy with user-friendly applications in Streamlit and Flask, supporting both image uploads and real-time webcam translations.

## Citations
- [“ASL Alphabet Dataset on Kaggle”](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/code?datasetId=23079&searchQuery=cnn)
- “Web Cam in a Webpage Using Flask and Python,” www.stackoverflow.com, 20 Feb. 2019.
- “Sign Language Detection with Python and Scikit Learn,” www.youtube.com, Computer Vision Engineer, 26 Jan. 2023.

## Team Members 

Rinad Almjishai (Repo Owner)

[Wael habib Alkiyani](https://github.com/Waelhab)

[Jawad Abdullah Sherbini](https://github.com/Jawadsherbini) 

[Albatul Ali Abusaq](https://github.com/Albatulabusaq)

[Luluh Khalid Alyahya](https://github.com/Luluh-Alyahya)

[Yasser Ahmed Alzahrani](https://github.com/Yasser-Alzahrani)

[Ali Abu Ali](https://github.com/aliabuali)

