This file contains some .py and models to train and do classification.  
To train Inception-Resnet-V2 for classification, run python Train_Classification.py   
In Classification_error.py , we use trained model to do classification and record the wrong choice  
In FP_all.log and FN_all.log we record image names for different error type  
We could visualize them from Main_Ship.ipynb to see the effect of classification.  

From following link we can get the trained model of Inception-Resnet-V2
Use torch.load() to load this model
https://drive.google.com/open?id=1Tn3X_HVsnjfwUMjFsCrfqUhspFFZI412
