import os
import pickle

import numpy as np
import tensorflow as tf

from utils import traiter_fichier, N_MFCC

# =========================================================
# PARAMÈTRES
# =========================================================

DOSSIER_MODELE = r"E:\Projet\Python\Exam_RNA\Modele"

CLASSES = ["Moi", "Personne", "Autre"]

# =========================================================
# FICHIER AUDIO À TESTER
# =========================================================

CHEMIN_AUDIO = r"E:\Projet\Python\Exam_RNA\Donnees\Test\Autre\Elizabeth test.wav"

# =========================================================
# CHARGEMENT DU MODÈLE ET DU SCALER
# =========================================================

modele = tf.keras.models.load_model(
    os.path.join(DOSSIER_MODELE, "modele_locuteur.keras")
)

with open(os.path.join(DOSSIER_MODELE, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# =========================================================
# TRAITEMENT DU FICHIER AUDIO
# =========================================================

vecteurs = traiter_fichier(CHEMIN_AUDIO, n_mfcc=N_MFCC)
vecteurs = np.array(vecteurs)

# Normalisation avec le MÊME scaler que l'entraînement
vecteurs = scaler.transform(vecteurs)

# =========================================================
# PRÉDICTION (moyenne des probabilités sur tous les segments)
# =========================================================

predictions = modele.predict(vecteurs)

print("\n===== PRÉDICTION PAR SEGMENT =====")
for i, proba in enumerate(predictions):
    classe_segment = CLASSES[np.argmax(proba)]
    print(f"Segment {i+1} : {proba} -> {classe_segment}")

probabilite_moyenne = np.mean(predictions, axis=0)
classe_predite = np.argmax(probabilite_moyenne)

# =========================================================
# MESSAGES PAR CLASSE
# =========================================================

MESSAGES = {
    "Moi": "C'est la voix de 'Jael'",
    "Personne": "C'est la voix de Nelson Mandela",
    "Autre": "C'est la voix d'une autre personne",
}

# =========================================================
# RÉSULTAT GLOBAL
# =========================================================

print("\n===== RÉSULTAT GLOBAL =====")
print("Probabilités moyennes :", probabilite_moyenne)

classe_finale = CLASSES[classe_predite]
print(MESSAGES[classe_finale])

# =========================================================
# DÉCISION D'ACCÈS
# =========================================================

if classe_finale == "Moi":
    print("\n✅ ACCÈS AUTORISÉ")
else:
    print("\n❌ ACCÈS REFUSÉ")