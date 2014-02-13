**************************MACHINE LEARNING PROJECT FALL 2013********************************
@Sudarshan
shetty.sud@gmail.com
Title: Sentimental Classification of IMDb Movie Reviews
Date: 12/14/2013
****************************************README**********************************************

I have used python programming/scripting language to code my implementations. 

	Python version: 2.7
	Supporting APIs: nltk (http://nltk.org/install.html)

Download the Dataset from this link(IMDb Dataset – Machine Learning Project ):
			http://edusid.com/blog/resources/
			
==============================DATASET DESCRIPTION================================

The Dataset folder has all the data needed to execute the code. Dataset folder has two 
sub-folders in it namely original and split. 

oringinal folder has the data that was obtained from the Cornell NLP site. Original folder 
has 3 folders in it,
	1. 1400_UnSeperated - all 1400 files 
	2. dataset_original - with 700 positive and 700 negatives files and also the README.
			      For more information on the dataset, you can have a look at the
			      README file in this folder.
	3. weka_dataset_original - the .arff format file of all 1400 reviews

split folder contains the different Datasets which I have used in my experimentations. 
	1. dataset1 folder has:
		train: 490 pos 490 neg 	
		test : 210 pos 210 neg
	2. dataset2 folder has:
		train: 630 pos 630 neg 	
		test : 70 pos 70 neg
	3. dataset3 folder has:
		train: 280 pos 280 neg 	
		test : 420 pos 420 neg

==============================BASELINE SYSTEM=================================
The Code folder has a folder called Preprocessing. This folder will provide you with a python 
script(preprocess.py) to implement the baseline system as mentioned in the Report. Also, the 
folder contains two files, negative and positive, which will give you the list of word used in
the baseline system implementation.

   To execute preprocess.py, first change the following file paths to your local system file paths:
pos_file_path = path of the file containg all positive words
neg_file_path = path of the file containg all positive words
files_path = path of the folder which contains all 1400 files(1400_UnSeperated)

   Once you have changed the filepaths: you can execute the program by typing: python preprocess.py

===============================NAVIE BAYES====================================
The Code folder has a folder called NaiveBayesProject. This folder will provide you with a python 
scripts to implement the Naive Bayes classifier as mentioned in the Report. 

    Execution Steps : python main.py

Supporting files: second.py

=============================LOGISTIC REGRESSION==============================
The Code folder has a folder called LogisticRegressionProject. This folder will provide you with 
a python scripts to implement the LR classifier as mentioned in the Report. 

Before executing LR change the following paths in main.py to your local system file paths:

   a. path_for_training_pos = this is the path of the training set pos folder
   b. path_for_training_neg = this is the path of the training set neg folder
   c. path_for_class = this is the path of the whole training set folder
   d. test_path_pos = this is the path of the test set pos folder
   e. test_path_neg = this is the path of the test set neg folder

Once you have changed the filepaths: you can execute the program by typing: python main.py

Supporting files: second.py
				  LR.py

=============================SUPPORT VECTOR MACHINES===========================
I have mad use of Weks Machine Learning tool for implementing SVM's. Weka has LibSVM classifier in it, 
we will make use of this classifier.

Weka accepts input file in .arff format or CSV format. Therefore we need to convert our text files into 
.arff format, we can do this by using the following command in Weka's Simple CLI:

	java weka.core.converters.TextDirectoryLoader -dir input_directory > output.arff

Once the conversion is done, we can use the Weka Explorer to load the file:

	Weka -> Explorer -> Preprocess -> Openfile

I have included the dataset.arff file in the weka_dataset_original folder located in original under Dataset.

The file loaded will still have an attribute called text in String format, we need to convert it to either 
nominal or numeric format which are supported by Weka's Classifiers. 

In the Preprocess tab:
	 
	Filter -> choose -> filters -> Unsupervised -> attribute -> StringToWordVector -> Ok
	Click the the tab next to choose button and set outputWordCounts: True -> Ok
	
and then click on Apply. This will convert the text attribute from String to numeric.

Next, move to Classify tab on Weka's Explorer:
	
	Classifier -> Choose -> functions -> LibSVM

You can change the KernelType or SVMType by clicking the options tab, located next to choose button and select 
various options available on LibSVM.

Once you have selected the KernelType and SVMType, we need to split the loaded file into training and testing, 
we can do this by using the percentage split option and specifying the split %age.

========================================END OF README===========================================================

