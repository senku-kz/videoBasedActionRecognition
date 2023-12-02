### UCF Center for Research in Computer Vision: 
https://www.crcv.ucf.edu/data/UCF_Sports_Action.php

### Data set:
https://www.crcv.ucf.edu/data/ucf_sports_actions.zip

### Download data set from UCF Center for Research in Computer Vision:
wget --no-check-certificate -c https://www.crcv.ucf.edu/data/ucf_sports_actions.zip


### Unzip downloaded file:
unzip ucf_sports_actions.zip

## Run TensorBoard
python -m tensorboard.main --logdir=Logs/


pip freeze > requirements.txt