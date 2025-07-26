This project focuses on classifying 30 different plant and tree species from images using deep learning. Leveraging EfficientNetB0 through transfer learning on top of a well-preprocessed dataset, the model aims to learn and distinguish between fine-grained categories of flora with high accuracy.

 Dataset Overview
Contains 30 tree/plant species, including neem, mango, bamboo, pipal, sugarcane, etc.

Folder structure follows:

dataset/
  ├── mango/
  ├── neem/
  ├── sugarcane/
  └── ...
Dataset contains class imbalance (e.g., other has more samples), which is tackled in the training process.

 Preprocessing Pipeline
1. Image Cleaning
Removed broken/corrupted images.
Ensured all images are of proper size and RGB format.

2. Normalization
Images were normalized by rescaling pixels from [0, 255] to [0, 1].

3. Data Augmentation
Implemented using ImageDataGenerator:

Random rotation

Horizontal flipping

Zoom transformations

ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
Handling Class Imbalance
Used collections.Counter to analyze class distribution.

Applied class_weight in model training to penalize dominant classes like 'other' and uplift underrepresented ones.
 Model Architecture
Final Model: Transfer Learning with EfficientNetB0

EfficientNetB0 (imagenet weights, unfrozen)
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(128, activation='relu')
→ Dropout(0.3)
→ Dense(30, activation='softmax')
Used Adam optimizer with learning_rate = 0.0001

Loss function: categorical_crossentropy

Training Setup
Framework: TensorFlow 2 / Keras

Platform: Google Colab

Accelerator: TPU / GPU

Epochs: 10 (can be scaled)

Batch Size: 32

Training/Validation split: 80/20

Results
Significant improvement in minority class prediction after applying class_weight.

The model generalizes well on validation set despite original class imbalance.

Training loss steadily decreases and validation accuracy stabilizes (see notebook plots).

Model Export
Final model saved as:

tree_species_classifier.h5
Can be reloaded using:

tf.keras.models.load_model('tree_species_classifier.h5')
Future Improvements
Use F1-score and per-class metrics for better performance diagnostics.

Visualize Grad-CAM heatmaps to understand CNN attention.

Apply stratified sampling or SMOTE for synthetic minority image generation.
Dependencies
bash
Copy
Edit
tensorflow
tensorflow-addons
numpy
matplotlib
sklearn
version: 1.0
How to Run

!pip install -U tensorflow
# Load and preprocess images
# Train model
# Save or evaluate model
🔗 Project Author
GitHub: Rhivu
Notebook: TreesClassification.ipynb
Model file: tree_species_classifier.h5
