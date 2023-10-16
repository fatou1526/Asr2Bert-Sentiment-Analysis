# Asr2Bert-Sentiment-Analysis
This repository defines github actions to construct a pipeline that load an audio (.wav) transcribes the audio using an ASR model and makes a sentiment analysis of the transcription using a binary classification Bert model trained using allocine movies review 

 Ce rapport est reparti en trois parties: 
 1. Mise en place d'un modèle ASR en français 
 2. Construction d'un modèle d'analyse de sentiment 
 3. Inférence (transcription de l'audio en texte et analyse de sentiment du texte transcrit) 
 
## Mise en place du modele ASR
Dans ce projet, il s'agit de choisir sur le hub de huggingface un modèle de ASR en français. Pour ce faire, de recherches et des tests ont été faits sur huggingface afin d'avoir un modèle capable de transcrire un audio français. Après plusieurs tests, nous avons choisi le modèle https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french qui a été finetuner à partir du modèle https://huggingface.co/facebook/wav2vec2-large-xlsr-53 avec les données en français de https://huggingface.co/datasets/common_voice.

## Construction du modele d'analyse de sentiment
Pour élaborer le modèle d'analyse de sentiment, nous avons utilisé les données disponibles à l'adresse suivante : https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews. Ces données correspondent à des critiques de films en langue française. Les données sont déjà repartie en train, test et validation datasets avec les tailles respectives de 80, 10, 10 en pourcentage. Par la suite, nous avons chargé le modèle BERT de "nlptown/bert-base-multilingual-uncased-sentiment" provenant de Hugging Face. Ce modèle a été affiné pour l'analyse de sentiment dans six langues différentes : anglais, allemand, néerlandais, espagnol, italien, et notamment le français. Il est employé dans le cadre de ce projet afin d'affiner un modèle d'analyse de sentiment basé sur une classification binaire, comprenant les catégories négative et positive.

Nous avons d'abord defini la variable config contenant tous les paramètres utilisés dans le modèle allant du chargement de données à l'entrainement du modèle, sa validation et son test.

Ensuite, la classe MyDataset est définie pour charger les données, prétraiter les données en séparant les textes et les labels puis transformant les textes en tokens afin de faciliter la compréhension des données par le modèle.

Nous avons fait appel à la classe DataLoader en définissant une fonction dataloader. Cette fonction permet de charger les données en batch après leur prétraitement. Cette fonction est appliquée pour les trois dataframes train, test, validation.

La classe SentimentAnalysisBertModel est définie pour représenter le modèle à affiner en se basant sur le modèle de hugging face et en définissant le nombre de classes. 

En les fonctions d'entrainement, de validation et de test ont été définies avant d'executer la fonction main qui charge les données, les prétraite, charge les données prétraitées par lots, entraine le modèle, l'évalue et le teste, affiche les métriques (loss et accuracy) de la validation et du test et enfin charge le modèle dans le hub de huggingface.

## Inference sur FastAPI
L'inférence est faite sur fastAPI. D'abord, dans le module utils.py nous avons chargé le modèle de ASR et avons créé une classe ASRInference contenant les fonctions nécessaires pour transcrire un fichier audio en texte. Ensuite, toujours dans utils.py, nous avons défini le dictionnaire de configuration config et la classe SentimentAnalysisBertModel pour traiter le texte en entrée dans le modèle d'analyse de sentiment. 

Enfin, dans main.py, nous avons fait appel à FastAPI app et avons défini les classes binaires negative et positive. Le modèle affiné pour l'analyse de sentiment est chargé et en utilisant une requête post et une fonction inference, la transcription de l'audio et son analyse de sentiment sont faites en une seule fois. Postman est utilisé pour faire de tests. Les resultats (captures des tests se trouvent dans le dossier captures de ce repertoire Github).
