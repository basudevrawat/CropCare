import os
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

def train_model():
    """
    Train the plant disease detection model using the downloaded dataset
    """
    print("Starting model training process...")
    
    # Check if dataset directories exist
    if not os.path.exists('train') or not os.path.exists('valid'):
        print("Error: Dataset directories not found.")
        print("Please make sure 'train' and 'valid' directories are in the current directory.")
        print("Download the dataset from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        return
    
    # Print info about dataset
    train_dir = 'train'
    valid_dir = 'valid'
    
    print(f"\nTraining data located in: {train_dir}")
    print(f"Validation data located in: {valid_dir}")
    
    # First, get the class names from both directories to check for mismatches
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
    valid_classes = [d for d in os.listdir(valid_dir) if os.path.isdir(os.path.join(valid_dir, d)) and not d.startswith('.')]
    
    print(f"\nFound {len(train_classes)} classes in training set")
    print(f"Found {len(valid_classes)} classes in validation set")
    
    # Find classes that are in training but not in validation
    missing_in_valid = set(train_classes) - set(valid_classes)
    if missing_in_valid:
        print(f"\nWARNING: Found {len(missing_in_valid)} classes in training set that are not in validation set:")
        for cls in missing_in_valid:
            print(f"  - {cls}")
        
        print("\nRemoving these classes from training set to ensure compatibility...")
        for cls in missing_in_valid:
            # Skip if the class doesn't exist (just to be safe)
            if os.path.exists(os.path.join(train_dir, cls)):
                print(f"  - Moving {cls} out of training set")
                # Create a backup directory if it doesn't exist
                os.makedirs('excluded_classes', exist_ok=True)
                # Move the directory
                shutil.move(os.path.join(train_dir, cls), os.path.join('excluded_classes', cls))
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    
    # Process training images
    training_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    
    # Process validation images
    validation_set = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    
    # Get class names
    class_names = validation_set.class_names
    print(f"\nDetected {len(class_names)} classes in the dataset.")
    
    # Save class names to file for later reference
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Build the CNN model
    print("\nBuilding CNN model...")
    
    # Create a sequential model with Input layer first
    inputs = tf.keras.Input(shape=(128, 128, 3))
    
    # First convolutional block
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Second convolutional block
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Third convolutional block
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Fourth convolutional block
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Fifth convolutional block
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Dropout and fully connected layers
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1500, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    
    # Create the model
    cnn = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Model summary
    cnn.summary()
    
    # Compile the model
    print("\nCompiling model...")
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Use standard Adam, not legacy
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training
    print("\nStarting training (this may take several hours)...")
    epochs = 10  # Adjust as needed
    
    # Add callback to save model after each epoch
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'trained_plant_disease_model_checkpoint.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    
    training_history = cnn.fit(
        x=training_set,
        validation_data=validation_set,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    
    # Save the final model
    print("\nSaving model...")
    cnn.save('trained_plant_disease_model.keras')
    
    # Save training history
    with open('training_hist.json', 'w') as f:
        json.dump({key: [float(x) for x in value] for key, value in training_history.history.items()}, f)
    
    # Plot training results
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_history.history['accuracy'], label='Training Accuracy', color='red')
    plt.plot(epochs_range, training_history.history['val_accuracy'], label='Validation Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_history.history['loss'], label='Training Loss', color='red')
    plt.plot(epochs_range, training_history.history['val_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    print("\nTraining complete!")
    print(f"Final training accuracy: {training_history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {training_history.history['val_accuracy'][-1]:.4f}")
    print("\nThe trained model has been saved as 'trained_plant_disease_model.keras'")
    print("You can now run the application with 'python app.py'")

if __name__ == "__main__":
    train_model()