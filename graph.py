import keras
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Charger  modèle
model = keras.models.load_model("model_ultimate_7000.keras")
taille_cible = (150,150)
def classify_image(img):
    if isinstance(img, np.ndarray):
        # Convertir numpy array en PIL Image
        img_pil = Image.fromarray(img.astype('uint8'))
    else:

        img_pil = img
    
    # Redimensionner image pour que ia reconnaise comme entrainer
    img_resized = img_pil.resize(taille_cible)
    

    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prédiction
    prediction = model.predict(img_array)
    

    if prediction.shape[1] == 1:
        score = float(prediction[0][0])
        
        if score < 0.5:
            # C'est la classe 0 (Femme)
            confidence = 1.0 - score
            return {"Femme": confidence, "Homme": score}
        else:
            # C'est la classe 1 (Homme)
            confidence = score
            return {"Homme": confidence, "Femme": 1.0 - score}
    
    else:
        labels = ["", "", ""] 
        return {labels[i]: float(prediction[0][i]) for i in range(len(labels))}

# 2. Interface mise à jour
demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3)
)

demo.launch(share=True)