# trichotillomania
A system for automated video therapy using machine learning to reduce unwanted body-focused behavior like trichotillomania requires a multi-step approach. 
The 3 programs provide the following functions

**Data Collection:**

Using webcams, capture videos of individuals displaying the unwanted behavior. This will form the "positive" class.
Capture videos without the behavior for the "negative" class.
Label these videos or frames accurately.
Data Preprocessing:

Convert videos into frames for ease of processing.
Use techniques like resizing, normalization, and data augmentation (rotations, flips) to enhance the dataset.
Model Building using TensorFlow/Keras:

Use Convolutional Neural Networks (CNNs) as they're effective for image/video data.
The architecture could have multiple convolutional layers, pooling layers, and fully connected layers, ending with a binary classification (behavior present or not).
Training:

Split the data into training and validation sets.
Train the CNN on the training set while validating its performance on the validation set.
Use techniques like dropout for regularization, and callbacks like early stopping to prevent overfitting.
Save the model that performs best on the validation data.
Detection in Real-time:

Using webcam feed, convert the video into frames in real-time.
Pass each frame through the trained model to detect the unwanted behavior.
If the behavior is detected, provide immediate feedback, which can be an alert or a sound.
Positive Reinforcement:

If the system detects the user refraining from the unwanted behavior for a specified duration, provide positive feedback. This can be in the form of encouraging messages or even rewards.
User Interface:

Develop a user-friendly interface that provides real-time feedback, shows progress over time, and offers resources for further assistance.
Ensure privacy concerns are addressed, as this is sensitive personal data.
Model Updating:

Periodically retrain the model with new data to improve accuracy.
Allow users to provide feedback on false positives/negatives to refine the model.
Deployment:

Ensure the application runs smoothly with real-time webcam feeds.
Optimize the model for latency to provide instant feedback.
