import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# PARAMÈTRES
# =========================================================

DOSSIER_DONNEES = r"E:\Projet\Python\Exam_RNA\Donnees"
DOSSIER_MODELE = r"E:\Projet\Python\Exam_RNA\Modele"

NB_CLASSES = 3          # Moi=0, Personne=1, Autre=2
BATCH_SIZE = 32
EPOCHS = 100

# =========================================================
# CHARGEMENT DU DATASET EXTRAIT
# =========================================================

print("===== CHARGEMENT DU DATASET =====")

X_train = np.load(os.path.join(DOSSIER_DONNEES, "X_train.npy"))
y_train = np.load(os.path.join(DOSSIER_DONNEES, "y_train.npy"))
X_test = np.load(os.path.join(DOSSIER_DONNEES, "X_test.npy"))
y_test = np.load(os.path.join(DOSSIER_DONNEES, "y_test.npy"))

print(f"X_train : {X_train.shape}   y_train : {y_train.shape}")
print(f"X_test  : {X_test.shape}   y_test  : {y_test.shape}")

# =========================================================
# NORMALISATION DES CARACTÉRISTIQUES (MFCC)
# =========================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# GESTION DU DÉSÉQUILIBRE DES CLASSES
# =========================================================

poids_classes = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
poids_classes = {i: poids for i, poids in enumerate(poids_classes)}

print("\nPoids de classes calculés :", poids_classes)

# =========================================================
# MODÈLE : MLP POUR 3 CLASSES DE VOIX
# =========================================================

nb_entrees = X_train.shape[1]   # = N_MFCC (défini dans utils.py)

modele = tf.keras.Sequential([

    layers.Input(shape=(nb_entrees,)),

    # Couches cachées : activation ReLU
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),

    # Couche de sortie : 3 neurones, activation softmax
    layers.Dense(NB_CLASSES, activation='softmax')
])

# =========================================================
# COMPILATION
# =========================================================

modele.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # labels = entiers (0,1,2)
    metrics=['accuracy']
)

print("\n===== RÉSUMÉ DU MODÈLE =====")
modele.summary()

# =========================================================
# ENTRAÎNEMENT
# =========================================================

print("\n===== ENTRAÎNEMENT =====")

historique = modele.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=poids_classes
)

# =========================================================
# ÉVALUATION
# =========================================================

print("\n===== ÉVALUATION =====")

loss, accuracy = modele.evaluate(X_test, y_test)

print(f"\nLoss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")

# =========================================================
# SAUVEGARDE (MODÈLE + SCALER)
# =========================================================

os.makedirs(DOSSIER_MODELE, exist_ok=True)

modele.save(os.path.join(DOSSIER_MODELE, "modele_locuteur.keras"))

with open(os.path.join(DOSSIER_MODELE, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print(f"\nModèle sauvegardé : {os.path.join(DOSSIER_MODELE, 'modele_locuteur.keras')}")
print(f"Scaler sauvegardé : {os.path.join(DOSSIER_MODELE, 'scaler.pkl')}")