import sys
from sklearn.cross_decomposition import PLSRegression,CCA,PLSCanonical
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from scipy import spatial

with open('word_vector.txt', 'r') as fi:
    lines = fi.readlines()

vector_dict = {}
wiki_datasheet = []
i = 0
for line in lines:
    i += 1
    temp_list = []
    words = (line.strip("\n")).split(" ")
    for w in words[1:]:
        if w == '':
            continue
        temp = float(w.strip('\n'))
        temp_list.append(temp)
    vector_dict[words[0]] = temp_list

with open('new_dataset.txt', 'r') as fw:
    wiki_lines = fw.readlines()

for wl in wiki_lines:
    wl_list = []
    wds = (wl.strip("\n")).split(",")
    for w in wds:
        wl_list.append(w)
    if not wl_list in wiki_datasheet:
        wiki_datasheet.append(wl_list)

x = []
y = []
x_n = []
y_n = []

train_end = len(wiki_datasheet)*0.8
test_start = len(wiki_datasheet) - train_end

itr = 0

pls = PLSRegression(n_components=2, scale=True, max_iter=1000, tol=0.1, copy=True)


for item in wiki_datasheet:
    if itr < train_end:
        y1 = item[0]
        x1 = item[1]
        x2 = item[2]
        temp1 = vector_dict[x1]
        temp2 = vector_dict[x2]
        temp3 = vector_dict[y1]

        temp4 = []
        for e in temp1[:]:
            temp4.append(e)
        for e in temp2[:]:
            temp4.append(e)
        if len(temp4) == 600 and len(temp3) == 300:
            x.append(temp4)
            y.append(temp3)
        else:
            print(y1,x1,x2)
        itr += 1
    else:
        y1 = item[0]
        x1 = item[1]
        x2 = item[2]
        temp1 = vector_dict[x1]
        temp2 = vector_dict[x2]
        temp3 = vector_dict[y1]

        temp4 = []
        for e in temp1[:]:
            temp4.append(e)
        for e in temp2[:]:
            temp4.append(e)
        if len(temp4) == 600 and len(temp3) == 300:
            x_n.append(temp4)
            y_n.append(temp3)


npx = np.asarray(x, dtype = np.float64)
npy = np.asarray(y, dtype = np.float64)

npxn = np.asarray(x_n, dtype = np.float64)
npyn = np.asarray(y_n, dtype = np.float64)
cca = PLSCanonical(n_components=2)
cca.fit_transform(npx, npy)
npx,npy = cca.transform(npx, npy)
npxn,npyn = cca.transform(npxn,npyn)


pls.fit(npx,npy)
params=pls.get_params(deep=True)
print(params)
pls.set_params(**params)

y_score = pls.predict(npxn)

sim_count = 0
tol = 0.1

for index in range(len(y_score)):
    sub_result = np.subtract(y_score, npyn)
    result = 1 - spatial.distance.cosine(y_score[index], npyn[index])
    print("similarity of test example "+str(index)+" = " + str(result))
    if (1 - math.fabs(result)) <= tol:
            sim_count += 1

print "Count of correct prediction = ", sim_count
acc = float(sim_count)/float(len(y_score))
print ("Accuracy = " + str(acc*100))
