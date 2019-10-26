# copier/coller la commande ci-dessous dans un terminal
# python3 prediction.py --model pokedex.model --labelbin lb.pickle --image exemples/akita_inu.jpg

# importer les fonctions nécessaires pour la classification
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construction des arguments avec le model d'entrainement, le label et l'image
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# chargement de l'image d'exemple
image = cv2.imread(args["image"])
output = image.copy()
 
# prétraitement de l'image
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# chargement du réseau de neurones entraîné 
print("Chargement du réseau neuronal...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# classification de l'image demandée
print("Classification de l'image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# comparaison du nom du fichier image avec la prédiction
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "Ok" if filename.rfind(label) != -1 else "Non Ok"

# construction de la classification afin de l'afficher sur l'image demandée
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=800)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (25, 102, 255), 2)

# affichage de l'image de sortie 
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
