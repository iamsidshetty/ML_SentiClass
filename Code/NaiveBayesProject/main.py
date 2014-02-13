__author__ = 'sid'

from second import *
import glob
import math

#Change the keyword "dataset" to any of the specified dataset
path_for_training_pos = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset1/train/pos'
path_for_training_neg = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset1/train/neg'
path_for_class = '/home/sid/Desktop/Project_ML/work/data/Dataset/dataset1/train'
test_path_pos = '/home/sid/Desktop/Project_ML/work/data/Dataset/dataset1/test/pos'
test_path_neg = '/home/sid/Desktop/Project_ML/work/data/Dataset/dataset1/test/neg'

No_of_posfiles_count = 0
No_of_negfiles_count = 0
No_of_class_count = 2

def GenerateFilePaths(path):
    fileAddressList = []
    for fileaddress in glob.glob(os.path.join(path, '*.txt')):
        fileAddressList.append(fileaddress)
    return fileAddressList

No_of_posfiles_count = Count_NumberOfFiles(path_for_training_pos)
No_of_negfiles_count = Count_NumberOfFiles(path_for_training_neg)
total_files = No_of_posfiles_count + No_of_negfiles_count

Vocabulary = []
posfiles = GenerateFilePaths(path_for_training_pos)
negfiles = GenerateFilePaths(path_for_training_neg)

#to generate vocab
for eachfile in posfiles:
    Generate_vocabulary(eachfile, Vocabulary)

for eachfile in negfiles:
    Generate_vocabulary(eachfile, Vocabulary)

#to generate the dictionary of postokens
postokens = {}
for eachfile in posfiles:
    concat(eachfile, postokens)

postokenslen = 0
for k, v in postokens.iteritems():
    postokenslen += v

#to generate the dictionary of negtokens
negtokens = {}
for eachfile in negfiles:
    concat(eachfile, negtokens)
negtokenslen = 0
for k, v in negtokens.iteritems():
    negtokenslen += v

classes = []
for dirname, dirnames, filenames in os.walk(path_for_class):

    for subdirname in dirnames:
        path = os.path.join(dirname, subdirname)
        classes.append(path)

#*********to find the training Conditional Probability, prior and Vocabulary************
total = 0
priors = {}
for i, eachclass in enumerate(classes):
    if i == 0:
        pathnames = eachclass
        No_of_files = len([item for item in os.listdir(pathnames) if os.path.isfile(os.path.join(pathnames, item))])
        prior = No_of_files / float(total_files)
        priors["pos"] = prior
        posConditionalProb = {}
        B = len(Vocabulary)
        for term in Vocabulary:
            if term in postokens:
                T_c_term = postokens.get(term)
                condprob = (T_c_term + 1) / float(postokenslen + B)
                posConditionalProb[term] = condprob
            else:
                condprob = (0 + 1) / float(postokenslen + B)
                posConditionalProb[term] = condprob

    if i == 1:
        pathnames = eachclass
        No_of_files = len([item for item in os.listdir(pathnames) if os.path.isfile(os.path.join(pathnames, item))])
        prior = No_of_files / float(total_files)
        priors["neg"] = prior
        negConditionalProb = {}
        B = len(Vocabulary)

        for term in Vocabulary:
            if term in negtokens:
                T_c_term = negtokens.get(term)
                condprob = (T_c_term + 1) / float(negtokenslen + B)
                negConditionalProb[term] = condprob
            else:
                condprob = (0 + 1) / float(negtokenslen + B)
                negConditionalProb[term] = condprob


#**************Apply NB to test docs*********************
#using: priors, Vocabulary, posConditionalProb, negConditionalProb, Classes

testposfiles = GenerateFilePaths(test_path_pos)
testnegfiles = GenerateFilePaths(test_path_neg)

#To generate the paths of training files
trainposfiles = GenerateFilePaths(path_for_training_pos)
trainnegfiles = GenerateFilePaths(path_for_training_neg)

#for pos test docs
correctclassification = 0
misclassification = 0
for eachfile in testposfiles:
    docdictionary = {}
    docdictionary = concat(eachfile, docdictionary)
    score_neg = math.log(priors.get('neg'), 10)
    score_pos = math.log(priors.get('pos'), 10)
    #for negclass
    for term in docdictionary:
        if term not in negConditionalProb:
            newcp = (0 + 1) / float(negtokenslen + B)
            times = docdictionary.get(term)
            score_neg += times * math.log(newcp, 10)
        else:
            times = docdictionary.get(term)
            score_neg += times * math.log(negConditionalProb.get(term), 10)
            #for posclass
    for term in docdictionary:
        if term not in posConditionalProb:
            newcp = (0 + 1) / float(postokenslen + B)
            times = docdictionary.get(term)
            score_pos += times * math.log(newcp, 10)
        else:
            times = docdictionary.get(term)
            score_pos += times * math.log(posConditionalProb.get(term), 10)

    if score_pos > score_neg:
        correctclassification += 1
    else:
        misclassification += 1

poscorrect = correctclassification
posIncorrect = misclassification

"""
print "\n"
print " NAIVE BAYES RESULTS - WITH BAG OF WORDS APPROACH"
print "**************************************************"
print "Results on Test pos Data"
print "No of docs correctly classified as pos: " + str(correctclassification)
print "No of docs incorrectly classified as neg: " + str(misclassification)
print "\n"
negAccuracy = correctclassification / float(len(testposfiles))
negAccuracy *= 100
print "pos Accuracy: {0}%".format(str(negAccuracy))
print "**************************************************"
"""
#for neg test docs
correctclassification = 0
misclassification = 0
for eachfile in testnegfiles:
    docdictionary = {}
    docdictionary = concat(eachfile, docdictionary)
    score_neg = math.log(priors.get('neg'), 10)
    score_pos = math.log(priors.get('pos'), 10)
    #for negclass
    for term in docdictionary:
        if term not in negConditionalProb:
            newcp = (0 + 1) / float(negtokenslen + B)
            times = docdictionary.get(term)
            score_neg += times * math.log(newcp, 10)
        else:
            times = docdictionary.get(term)
            score_neg += times * math.log(negConditionalProb.get(term), 10)
            #for posclass
    for term in docdictionary:
        if term not in posConditionalProb:
            newcp = (0 + 1) / float(postokenslen + B)
            times = docdictionary.get(term)
            score_pos += times * math.log(newcp, 10)
        else:
            times = docdictionary.get(term)
            score_pos += times * math.log(posConditionalProb.get(term), 10)
    if score_neg > score_pos:
        correctclassification += 1
    else:
        misclassification += 1

negcorrect = correctclassification
negIncorrect = misclassification
"""
print "**************************************************"
print "Results on Test neg Data"
print "No of docs correctly classified as neg: " + str(correctclassification)
print "No of docs incorrectly classified as neg: " + str(misclassification)
print "\n"
negAccuracy = correctclassification / float(len(testnegfiles))
negAccuracy *= 100
print "neg Accuracy: " + str(negAccuracy) + "%"
print "**************************************************"
"""

print "\n=======================SUMMARY==========================\n"
print "Total Instances: " + str(len(testposfiles) + len(testnegfiles))
print "Correctly Classified Instances: " + str(poscorrect + negcorrect),

print "Accuracy obtained: " + str("{0:.2f}".format(((float(poscorrect) + negcorrect)/float((len(testposfiles) + len(testnegfiles)))) * 100)) + "%"
print "InCorrectly Classified Instances: " + str(posIncorrect + negIncorrect),
print "Accuracy obtained: " + str("{0:.2f}".format(((float(posIncorrect) + negIncorrect)/float((len(testposfiles) + len(testnegfiles)))) * 100)) + "%\n"