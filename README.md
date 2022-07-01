# Installation du système

1- Télécharger le Epic Game Launcher [ici](https://store.epicgames.com/fr/download)
2 - Installer Unreal Engine 4.27 via le Epic Game Launcher
3 - Redémarer le launcher Epic Games afin d'associer les fichiers .uproject à Unreal Engine. Si un message apparaît indiquant que les fichiers ne sont pas associés, cliquer sur "Fix it"

4 - Installer [Visual Studio 2022](https://visualstudio.microsoft.com/fr/vs/)
5 - Cloner le repo github *add link*
6 - Dans le dossier obtenu, déplacer le dossier 'Airsim' (contenant un fichier 'settings.json') dans les documents utilisateurs 

# Installation des modules Python
pip install gym
pip install msgpack-rpc-python
pip install airsim
pip install Pillow
pip install stable-baselines3
pip install tensorboard

# Lancement de la simulation
Le dossier Unreal contient deux environnements appellés 'Circuit Rond' et 'Circuit Test'. Chacun de ces environnements peut être lancé en suivant la même procédure.
Il est possible de procéder de deux manières. 
* Ouvrir le fichier 'Blocks.uproject' ouvre le projet dans l'éditeur Unreal Engine 4 et permet la modification du circuit.
* Exécuter le fichier 'WindowsNoEditor/Blocks.exe' lance la simulation sans éditeur. Cela permet d'améliorer les performances.

# Exécution des scripts python
Les différents scripts Python utilisés se trouvent dans le dossier 'PythonClient/dqn_car'.
## Phase d'apprentissage
L'apprentissage est lancé par le script 'dqn_car.py'. Deux réglages sont possibles, via une variable interne du script. 
* Si l'on règle `Load = False` alors un nouveau modèle est créé avant le démarage de l'entrainement
* Si l'on règle `Load = True` alors un ancien modèle est chargé selon le chemin d'accès spécifié. L'entrainement continu alors sur ce modèle
## Phase d'inférence