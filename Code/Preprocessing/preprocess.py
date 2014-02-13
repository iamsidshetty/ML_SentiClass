__author__ = 'sid'

import nltk
import os

pos_file_path = '/home/sid/Desktop/Project_ML/work/Preprocessing/positive'
neg_file_path = '/home/sid/Desktop/Project_ML/work/Preprocessing/negative'
files_path = '/home/sid/Desktop/Project_ML/work/Dataset/original/1400_UnSeperated/'
#files_path = '/home/sid/Desktop/Project_ML/work/data/original/dataset1/pos/'

file_content = open(pos_file_path).read()
pos_tokens = nltk.word_tokenize(file_content)

file_content = open(neg_file_path).read()
neg_tokens = nltk.word_tokenize(file_content)

positive = 0
negative = 0
ties = 0

dirList = os.listdir(files_path)
for fname in dirList:
    fname = ''.join([files_path, fname])
    file_content = open(fname).read()
    file_tokens = nltk.word_tokenize(file_content)
    file_tokens_lower = [item.lower() for item in file_tokens]
    pos_words_count = 0
    neg_words_count = 0
    for word in pos_tokens:
        if word in file_tokens_lower:
            pos_words_count += 1
    for word in neg_tokens:
        if word in file_tokens_lower:
            neg_words_count += 1
    if pos_words_count > neg_words_count:
        positive += 1
    elif neg_words_count > pos_words_count:
        negative += 1
    else:
        ties += 1

print "\n****************Baseline Results*******************"
print "No. of Positive Reviews: " + str(positive)
print "No. of Negative Reviews: " + str(negative)
print "No. of Ties(Both positive and negative): " + str(ties)

print "\nAccuracy obtained: " + str((float(positive + negative) / len(dirList)) * 100) + "%\n"