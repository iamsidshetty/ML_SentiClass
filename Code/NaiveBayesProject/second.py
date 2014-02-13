__author__ = 'sid'
import glob
import os

def Count_NumberOfFiles(path):
    No_of_files_count = 0
    for eachfile in glob.glob(os.path.join(path, '*.txt')):
        No_of_files_count += 1
    return No_of_files_count

#To generate the tokens of a class
def concat(path, classtokens):
    f = open(path, 'r')
    fileContent = f.read()
    for word in fileContent.split():
        if word in classtokens:
            occurence = classtokens.get(word)
            occurence += 1
            classtokens[word] = int(occurence)
        else:
            classtokens[word] = int(1)
    return classtokens

def Generate_vocabulary(filePath, Vocabulary):
    f = open(filePath, 'r')
    fileContent = f.read()
    for word in fileContent.split():
        if word in Vocabulary:
            pass
        else:
            Vocabulary.append(word)
    return Vocabulary



