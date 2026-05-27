import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
import datetime 
data_dir = "dataset/train"
img_size = (150, 150) # TAille des images
batch_size = 64       

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123,
    image_size=img_size, batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=img_size, batch_size=batch_size)


train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.Sequential([
    
    layers.Input(shape=(150, 150, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Rescaling(1./255),

    # --- BLOC 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # Couche de neurone pour reflechir
    layers.BatchNormalization(), 
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # Couche de neurone pour reflechir
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2), 


    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.4),


    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)), # Couche qui envoie le resulta
    layers.Dropout(0.5), 
    
    layers.Dense(1, activation='sigmoid') #Retourne avec sigmoid 0 ou 1
])

# --- OPTIMISATION INTELLIGENTE ---
#SI pas de amelioration divise vitesse
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       # On divise la vitesse par 2
    patience=3,       # Si pas d'amélioration pendant 3 tours
    min_lr=1e-6,      # Vitesse minimale
    verbose=1
)

# SI ya pas d'amelioration pendant 10 epoch Aretter et sauvgarder le model
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=0 # Permet de voir l'évolution des poids du réseau
)
# -------------------------------------------

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. ENTRAÎNEMENT ---
print(f"Lancement de la bête ")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    # --- NOUVEAU : AJOUT DU CALLBACK TENSORBOARD DANS LA LISTE ---
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback] 
)

# Sauvegarde finale
model.save('model_ultimate_7000.keras')