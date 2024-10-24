# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt 

from errordetector.dataset import *
from errordetector.utils.utils import *

from sys import argv


file_label = argv[1]
file_pred = argv[2]

label = h5read(file_label)
pred = h5read(file_pred)


accuracy, precision, recall, F1, thresholds = stat_summary(label, pred, np.arange(0,1.05,0.05))


for i in range(len(thresholds)):
	print('Thr = ' + str(np.round(thresholds[i],3)) + ', Precision = ' + str(np.round(precision[i],3)) + ', Recall = ' + str(np.round(recall[i],3)) + ', F1 = ' + str(np.round(F1[i],3)) + ', Accuracy = ' + str(np.round(accuracy[i],3)))
	

plt.figure()
plt.plot(recall, precision, '-o')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.grid()

plt.show()