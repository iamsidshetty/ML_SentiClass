from LR import *

   
No_of_pos_count = 0
No_of_neg_count= 0
No_of_class_count = 2
#original
path_for_training_pos = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset3/train/pos'
path_for_training_neg = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset3/train/neg'

path_for_class = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset3/train'

test_path_pos = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset3/test/pos'
test_path_neg = '/home/sid/Desktop/Project_ML/work/Dataset/split/dataset3/test/neg'


def GenerateFilePaths(path):
    fileAddressList = []
    for fileaddress in glob.glob( os.path.join(path, '*.txt') ):
        fileAddressList.append(fileaddress)
    return fileAddressList

No_of_pos_count = Count_NumberOfFiles(path_for_training_pos)
No_of_neg_count= Count_NumberOfFiles(path_for_training_neg)
total_files = No_of_pos_count + No_of_neg_count

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

testposfiles = GenerateFilePaths(test_path_pos)

testnegfiles = GenerateFilePaths(test_path_neg)


#To generate the paths of training files
trainposfiles = GenerateFilePaths(path_for_training_pos)
trainnegfiles = GenerateFilePaths(path_for_training_neg)


#Logistic Regression --> logRegress.py
#print "Do you want to Implement the MCAP LR Algorithm? (y/n): ",
#resp = raw_input()
#if resp == 'y':
print "\n******************Implementing LR***********************"
wgt = []
wgt = logisticRegression(Vocabulary, trainposfiles, trainnegfiles)
#print wgt
logistictesting(Vocabulary, wgt, testposfiles, testnegfiles)
#else:
#    pass
