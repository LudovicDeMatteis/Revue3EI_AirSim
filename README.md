# Introduction

Ce dépôt est associé à l'article "Apprentissage par renforcement de la conduite d’un véhicule sur AirSim" du numéro de Juillet 2022 de la revue 3EI.
Il est basé sur le dépôt [AirSim](https://github.com/microsoft/AirSim) puis réduit pour ne garder que les éléments essentiels à l'article.

# Installation du système

1. Télécharger le Epic Game Launcher [ici](https://store.epicgames.com/fr/download)

2. Installer Unreal Engine 4.27 via le Epic Game Launcher

3. Redémarer le launcher Epic Games afin d'associer les fichiers .uproject à Unreal Engine. Si un message apparaît indiquant que les fichiers ne sont pas associés, cliquer sur "Fix it"

4. Installer [Visual Studio 2022](https://visualstudio.microsoft.com/fr/vs/)

5. Cloner le [repo github](https://github.com/LudovicDeMatteis/Revue3EI_AirSim) 

6. Dans le dossier obtenu, déplacer le dossier 'Airsim' (contenant un fichier 'settings.json') dans les documents utilisateurs 

# Installation des modules Python
`pip install gym`

`pip install msgpack-rpc-python`

`pip install airsim`

`pip install Pillow`

`pip install stable-baselines3`

`pip install tensorboard`

# Lancement de la simulation
Le dossier Unreal contient deux environnements appellés **'Circuit Rond'** et **'Circuit Test'**. Chacun de ces environnements peut être lancé en suivant la même procédure.
Il est possible de procéder de deux manières. 
* Ouvrir le fichier `Blocks.uproject` ouvre le projet dans l'éditeur Unreal Engine 4 et permet la modification du circuit.
* Exécuter le fichier `WindowsNoEditor/Blocks.exe` lance la simulation sans éditeur. Cela permet d'améliorer les performances.

# Exécution des scripts python
Les différents scripts Python utilisés se trouvent dans le dossier `PythonClient/dqn_car`.
## Phase d'apprentissage
L'apprentissage est lancé par le script `dqn_car.py`. Deux réglages sont possibles, via une variable interne du script. 
* Si l'on règle `Load = False` alors un nouveau modèle est créé avant le démarage de l'entrainement
* Si l'on règle `Load = True` alors un ancien modèle est chargé selon le chemin d'accès spécifié. L'entraînement continue alors sur ce modèle

## Phase d'inférence
L'observation des résultats de l'agent se fait en lançant la simulation d'un des deux circuits puis en exécutant le script `dqn_car_test.py`. Dans ce script il est nécessaire de préciser le modèle à utiliser. Quelques modèles sont disponibles dans le dossier `Modeles` (le dossier *Saves* contient des modèles en vrac créés lors de nombreux tests).

Une fois ce script lancé, le véhicule doit commencer à évoluer dans le circuit.

# Modèles pré-entraînés
Plusieurs modèles pré-entraînés sont disponibles dans le dossier `PythonClient/dqn_car/Modeles`. Le fichier *modele_base* correspond à un modèle ne pouvant pas faire un tour de circuit mais ayant déjà les bases de l'apprentissage. Il sert de base pour des modèles plus complexes, permettant un entrainement moins long. Le *modele1* correspond à un entrainement supplémentaire réalisé sur cette base. La voiture est ici capable de faire plusieurs tours de circuits.
