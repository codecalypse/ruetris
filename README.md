# Ruetris
Ruetris is the machine learning enviroment developed for my master thesis. This enviroment is in the src-folder.
This repository consists of not only the enviroment, instead, provides networks for reinforcement learning, agents,
training routines, manual mode, test routines and evaluated data from my thesis. The thesis itself is also contained, but 
is currently only available in german (I have in mind to translate the work later on). The videos linked in the thesis are
on youtube: https://www.youtube.com/hashtag/mymasterthesis , the result files are on a separate repository: https://github.com/codecalypse/ruetris_results .
The simulation application showed on youtube and in thesis is not in the repository, because it is not my property. Enything else 
I did in my sparetime and shall be provided under the MIT lizenz to everyone, maybe someone can find it usefull.
At the end I want to mention, that this was my very first time doing something with neuronal networks and creating something 
bigger as simple functions, so please be indulgent and hold that in mind. Neverless I enjoyed the challange.

Usage:
1) Create the enviroment with the yml-file: conda env create -f env.yml
2) Activate the enviroment: conda activate ruetris
3) cd into the Folder: cd /some/path/ruetris
4) Open the train.py file to change hyperparameters: <preferded editor> train.py
5) You can alos type: python train.py -h for a list of parameters and chagne them from cli instead of editing the file.
6) Start the script from the cli: python train.py <change some parameters from cli>
7) To inspect training-graphs use tensorbaordX in another tab: tensorboard ./tensorboard

After the training has started, a train_config.txt and a summary.txt file should have been created in the main-folder.
After every save intervall a model is saved in the trained_model-folder. With the train_config.txt and the desired filename of the
model (one of maybe many in the trained_models-folder), the test_model.py script can be adapted and executed. For manual interaction, adapt
the manual_mode.py script and execute it from cli. Due to the library used for keyboard readings, the manual_mode.py-file has to be startet with sudo rights.
