# Abstract Detector

Abstract Detector est un projet de classification d'image par le Machine Learning. Il implémente un réseau neuronal convolutionnel (CNN) simple qui est entraîné à distinguer les images abstraites des images avec des élements reconnaissables. Il s'agit d'un pipeline complet d'entraînement, puis évaluation et enfin déploiement minimal

## Dataset

Le projet utilise kagglehub pour récupérer automatiquement les datasets :
- Un dataset de photographies, composé de "pankajkumar2002/random-image-sample-dataset" (photographies de nature) et de "ahmadahmadzada/images2000" (photographies d'êtres humains)
- Un dataset de peintures abstraites, composé d'un subset de "steubk/wikiart", sélectionnant les peintures de genre "Abstract_Expressionism", "Action_painting" et "Color_Field_Painting".

Le dataset est équilibré pour être parfaitement divisé en deux, et fait une taille totale de 8990. Il est divisé en un dataset d'entraînement (train_dataset) et un dataset de test (test_dataset).

## Model

Le modèle est un CNN simple, composé de :
- 2 blocs convolutionnels enchaînés
- 2 blocs linéaires
- Une fonction sigmoid en sortie, pour classification binaire.
Le tout agrémenté de fonctions ReLU d'activations et de downsamplings.

## Installation

```bash
git clone <URL_DU_REPO>
cd abstract_image_detector_ml

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Entraînement

```bash
python src/train.py
```
Les hyperparamètres sont configurés dans train_model.py.
Le modèle et le dataset de test sont sauvegardés dans models/


## Evaluation

```bash
python src/evaluate.py
```
Evalue l'accuraycy sur le dataset de test sauvegardé dans models/

## Déploiement

Une API RESTful minimale via Flask est accessible

```bash
python src/deploy.py
curl -X POST -F "file=@FILE.jpg" http://127.0.0.1:5000/predict
```
