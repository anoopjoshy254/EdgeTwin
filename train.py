import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse

# Import our custom data pipeline
from data_pipeline import process_data


def build_tiny_model(input_dim):
    """
    Builds an edge-deployable neural network with improved accuracy.

    Architecture:
        Input → Dense(32) → BatchNorm → Dropout(0.3)
               → Dense(16) → BatchNorm → Dropout(0.2)
               → Dense(1, sigmoid)

    Stays compatible with INT8 TFLite quantization for ESP32 / MCU deployment.
    BatchNorm layers are folded into preceding Dense layers at inference time,
    adding negligible size overhead to the final .tflite model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        # --- Hidden Layer 1 ---
        tf.keras.layers.Dense(32, use_bias=False, name='dense_1'),
        tf.keras.layers.BatchNormalization(name='bn_1'),
        tf.keras.layers.Activation('relu', name='relu_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),

        # --- Hidden Layer 2 ---
        tf.keras.layers.Dense(16, use_bias=False, name='dense_2'),
        tf.keras.layers.BatchNormalization(name='bn_2'),
        tf.keras.layers.Activation('relu', name='relu_2'),
        tf.keras.layers.Dropout(0.2, name='dropout_2'),

        # --- Output ---
        tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model


def quantize_model(keras_model, X_train, models_dir):
    """
    Converts a Keras model to Standard TFLite and INT8 Quantized TFLite.
    """
    print("[*] Converting to standard TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    std_tflite_path = os.path.join(models_dir, 'model.tflite')
    with open(std_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[*] Standard TFLite saved to {std_tflite_path} ({len(tflite_model)} bytes)")

    print("[*] Converting to INT8 Quantized TFLite...")

    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    quant_tflite_path = os.path.join(models_dir, 'model_quant.tflite')
    with open(quant_tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"[*] Quantized INT8 TFLite saved to {quant_tflite_path} ({len(tflite_quant_model)} bytes)")


def train_and_export(csv_path):
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Process & augment data
    X_train, X_test, y_train, y_test = process_data(csv_path, models_dir)
    input_dim = X_train.shape[1]

    # ------------------------------------------------------------------
    # Compute class weights to handle any residual imbalance
    # ------------------------------------------------------------------
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=unique_classes,
                                   y=y_train)
    class_weight_dict = dict(zip(unique_classes.astype(int), weights))
    print(f"\n[*] Class weights: {class_weight_dict}")

    print(f"\n[*] Building model with input dimension: {input_dim}")
    model = build_tiny_model(input_dim)
    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    best_model_path = os.path.join(models_dir, 'best_model.keras')

    callbacks = [
        # Stop early if val_loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        # Save the best checkpoint automatically
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Cosine annealing learning rate schedule
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: (
                lr * 0.5 * (1 + np.cos(np.pi * (epoch % 30) / 30))
                if epoch > 0 and epoch % 30 == 0 else lr
            ),
            verbose=0
        ),
        # Also keep ReduceLROnPlateau as secondary safety net
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n[*] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1,
        callbacks=callbacks
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n[*] Evaluating on Test Set...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy:  {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall:    {results[3]:.4f}")
    print(f"Test AUC-ROC:   {results[4]:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------
    h5_path = os.path.join(models_dir, 'model.h5')
    model.save(h5_path)
    print(f"\n[*] Keras model saved to {h5_path} ({os.path.getsize(h5_path)} bytes)")

    # Quantize for edge deployment
    quantize_model(model, X_train, models_dir)

    print("\n[*] Complete! Model trained with enhanced augmentation and improved architecture.")
    print(f"[*] Best checkpoint saved to: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TinyML Machine Failure model")
    parser.add_argument("--csv", type=str, required=True, help="Path to the training CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Dataset '{args.csv}' not found.")
        sys.exit(1)

    train_and_export(args.csv)
