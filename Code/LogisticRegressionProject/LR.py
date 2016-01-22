__author__ = 'sid'

import numpy
import scipy.sparse as sps
import random
from second import *
import math


def logisticRegression(Vocabulary, pFilespath, nFilespath):
    rows = len(pFilespath) + len(nFilespath)
    total_columns = len(Vocabulary)

    wo = 9999999999   #pos
    w1 = 1111111111   #neg

    listoflists = []
    for eachfile in pFilespath:
        v1 = []
        v1 = Generate_vocabulary(eachfile, v1)
        v1.append(wo)
        listoflists.append(v1)

    for eachfile in nFilespath:
        v1 = []
        v1 = Generate_vocabulary(eachfile, v1)

        v1.append(w1)
        listoflists.append(v1)

    pr = []
    for i in range(0, rows):
        p = random.random()
        pr.append(p)

    w = []
    for i in range(0, (total_columns)):
        w_i = random.random()
        w.append(w_i)
    oldwgts = [0] * total_columns

    ##________Main_________

    for iterat in range(0, 100):
        for e, eachexample in enumerate(listoflists):
            wixi = 0
            for it, word in enumerate(Vocabulary):
                if word in eachexample:
                    wixi += w[it] * 1

            pr[e] = 1 / (1 + math.exp(w[0] + wixi))

        dw = [0] * total_columns

        for ind, wgt in enumerate(w):
            for ec, eachfile in enumerate(listoflists):
                if Vocabulary[ind] in eachfile:
                    if w1 in eachfile:
                        dw[ind] += (-pr[ec])
                    else:
                        dw[ind] += (1 - pr[ec])

        for every, wt in enumerate(oldwgts):
            oldwgts[every] = w[every]

        for i, wt in enumerate(w):
            w[i] += 0.0001 * (dw[i] - (0.001 * float(w[i])))

        #To count the number of different wgts
        count = 0
        for i, wt in enumerate(oldwgts):
            diff = w[i] - float(oldwgts[i])

            if abs(diff) > 0.005:
                count += 1

        #print "Iteration: " + str(iterat + 1) + " Done!!"
    return w

    #    if count < 10:
    #        return w
    #    else:
    #        if iterat == 5:
    #            return w


def logistictesting(Vocabulary, wgt, testpos, testneg):
    wo = 9999999999   #pos
    w1 = 1111111111   #neg
    rows = len(testneg) + len(testpos)

    listoflistsfortest = []
    for eachfile in testpos:
        v1 = []
        v1 = Generate_vocabulary(eachfile, v1)
        v1.append(wo)
        listoflistsfortest.append(v1)

    for eachfile in testneg:
        v1 = []
        v1 = Generate_vocabulary(eachfile, v1)

        v1.append(w1)
        listoflistsfortest.append(v1)

    Correct = 0
    NotCorrect = 0

    prtest = [0] * rows
    for e, eachexample in enumerate(listoflistsfortest):
        Wixi = 0
        for indx, eachword in enumerate(Vocabulary):
            if eachword in eachexample:
                Wixi += wgt[indx] * 1

        prtest[e] = 1 / (1 + math.exp(wgt[0] + Wixi))
        Dr = 1 - float(prtest[e])
        pr = prtest[e] / Dr
        if pr < 0.5:
            if w1 in eachexample:
                Correct += 1
            else:
                NotCorrect += 1
        else:
            if wo in eachexample:
                Correct += 1
            else:
                NotCorrect += 1

    print "     LR RESULTS - WITH BAG OF WORDS APPROACH      "
    print "**************************************************"
    accuracy = (Correct / float(rows)) * 100
    print "Accuracy: " + str(accuracy) + "%"

    print "**************************************************"

