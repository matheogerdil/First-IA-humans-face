# 🧠 Gender Recognition AI (Custom CNN)

> Une Intelligence Artificielle de classification d'images capable de distinguer les genres (Homme/Femme) avec une précision de **~95%**, entraînée **"from scratch"** sur un dataset de 7000 images d'homme et de femme.

---

## 📋 Table des Matières
- [Aperçu du Projet](#-aperçu-du-projet)
- [Architecture du Modèle](#-architecture-du-modèle)
- [Performance & Résultats](#-performance--résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Dataset](#-structure-du-dataset)
- [Auteur](#-auteur)

---

## 🔭 Aperçu du Projet

Ce projet est une implémentation d'un **Réseau de Neurones Convolutif ** profond utilisant TensorFlow et Keras. L'objectif est de construire un modèle performant sans utiliser de modèles pré-entraînés, afin de maîtriser les concepts fondamentaux du Deep Learning :

* Convolution & Pooling
* Batch Normalization
* Data Augmentation
* Ordonnancement du taux d'apprentissage
* Prévention du surapprentissage

---

## 🏗 Architecture du Modèle

Le modèle utilise une architecture personnalisée de type **"Mini-VGG"**, optimisée pour la performance et la généralisation.

| Étage | Type de Couche | Détails Techniques |
| :--- | :--- | :--- |
| **Entrée** | Input | Image (150, 150, 3) + Augmentation |
| **Bloc 1** | Double Convolution | 32 filtres + BatchNormalization |
| **Bloc 2** | Double Convolution | 64 filtres + BatchNormalization |
| **Bloc 3** | Double Convolution | 128 filtres + BatchNormalization |
| **Bloc 4** | Convolution Profonde | 256 filtres + BatchNormalization |
| **Classification** | GlobalAveragePooling | Remplacement du "Flatten" pour réduire les paramètres |
| **Sortie** | Dense (Sigmoid) | Classification Binaire (0=Femme, 1=Homme) |

### Technologies Clés
* **Optimiseur :** Adam avec `ReduceLROnPlateau`.
* **Fonction de Perte :** Binary Crossentropy.
* **Callbacks :** EarlyStopping & ModelCheckpoint.

---

## 📊 Performance & Résultats

Le modèle a été entraîné sur environ **7 000 images** (3 500 Hommes / 3 500 Femmes) réparties en 80% Entraînement et 20% Validation.

* **Précision Finale (Validation) :** `94.74%` 🎯
* **Perte (Loss) :** `0.19`
* **Temps d'entraînement :** ~45 époques (Arrêt automatique via EarlyStopping).

> **Note :** L'écart entre l'entraînement (97%) et la validation (95%) est minime (< 3%), ce qui démontre une excellente généralisation et une absence de surapprentissage majeur.

---

## 💻 Installation

Clonez ce dépôt et installez les dépendances nécessaires :

```bash
git clone [https://github.com/VOTRE_NOM_UTILISATEUR/NOM_DU_REPO.git](https://github.com/VOTRE_NOM_UTILISATEUR/NOM_DU_REPO.git)
cd NOM_DU_REPO
pip install tensorflow numpy matplotlib
