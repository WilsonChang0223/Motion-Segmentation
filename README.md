# Motion Segmentation (DAVIS2016)
# Model
## v1 
It is pure pspnet. input is one frame
## v2 
I mimic the "Learning Video Object Segmentation with Visual Memory" paper. There are three models. The input of the semantic segmentation model is the last frame in three consecutive frame, and the input of the motion model is three consecitive frames. Finally, I concatenate the two output and enter the last model.
## v3
There are two models. The three consecutive frames will be calculated by the segmentation model frame by frame. I concatenate three of output and it would be calculated by the last model.

# How to use
1. Download DAVIS2016 dataset
2. unzip them and put them in the same Director
3. move the train_seq.txt and val_seq.txt into the DAVIS folder
4. assign the DAVIS path to the db_root_dir in Davis2pkl.py file
5. execute Davis2pkl.py and it will create a file called "davis.pkl"


