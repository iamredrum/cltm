print(__doc__)
import numpy as np
import scipy.io as sio
import sklearn
from sklearn.svm import SVR
import csv
num_sample = 40000
### read data 
#data_dir = '/Users/tejaswi/Desktop/LatentTree/coco_experiments/imagenet_coco_features/'
data_dir = '/home/teja/imagenet_coco_features/'
import h5py
mat_contents = h5py.File(data_dir+'images_all.probs.mat')
input_feat = mat_contents['scores'][()]
## extract input_features
reader=csv.reader(open(data_dir+'labels.csv',"rb"),delimiter=',')
x=list(reader)
labels=np.array(x).astype('int')

reader=csv.reader(open(data_dir+'indoor_filenames.csv',"rb"),delimiter=',')
x=list(reader)
indoor_samples=np.array(x).astype('int')
indoor_samples=indoor_samples.transpose()[0] -1

reader=csv.reader(open(data_dir+'label_header.csv',"rb"),delimiter=',')
x=list(reader)
label_header=np.transpose(np.array(x).astype('int'))
labels = labels[:,label_header.nonzero()[1]]
random_sample=np.random.randint(indoor_samples.shape[0],size=num_sample)
#random_sample = indoor_samples[random_sample];


input_sample = input_feat[random_sample,:]
label_sample = labels[random_sample,:]
gamma = .01
kernel_mat = sklearn.metrics.pairwise.rbf_kernel(input_sample,  gamma=.0000005)
lam = 1e-5
kernel_inverse = np.linalg.inv(kernel_mat + np.multiply(lam, np.identity(num_sample)))

multi_array_11 = np.zeros((num_sample,80,80))
multi_array_10 = np.zeros((num_sample,80,80))
multi_array_01 = np.zeros((num_sample,80,80))
multi_array_00 = np.zeros((num_sample,80,80))
for i in range(0,80):
        for j in range(0,80):
  		multi_array_11[:,i,j]  = np.multiply(label_sample[:,i],label_sample[:,j])
                multi_array_10[:,i,j]  = np.multiply(label_sample[:,i],np.subtract(1,label_sample[:,j]))
                multi_array_01[:,i,j]  = np.multiply(np.subtract(1,label_sample[:,i]),label_sample[:,j])
                multi_array_00[:,i,j]  = np.multiply(np.subtract(1,label_sample[:,i]),np.subtract(1,label_sample[:,j]))
eps = 0 
def calculateDistance(n,k):
        a = np.zeros((n,80,80))
        b = np.zeros((n,80,80))
        c = np.zeros((n,80,80))
        d = np.zeros((n,80,80))
        y = np.zeros((n,80,80))
        kernel_2 = np.dot(kernel_inverse,kernel_mat[:,n*k:n*(k+1)])
        for i in range(0,80):
                for j in range(0,80):
                        p11  = np.dot(np.transpose(multi_array_11[:,i,j]), kernel_2)
                        p00  = np.dot(np.transpose(multi_array_00[:,i,j]), kernel_2)
                        p01  = np.dot(np.transpose(multi_array_01[:,i,j]), kernel_2)
                        p10  = np.dot(np.transpose(multi_array_10[:,i,j]), kernel_2)
                        norm = p11+p00+p01+p10
                        a[:,i,j] = p00/norm
                        b[:,i,j] = p10/norm
                        c[:,i,j] = p01/norm
                        d[:,i,j] = p11/norm
                        abs_matrix =  np.absolute(np.subtract(np.multiply(a,d),np.multiply(b,c)) ) +eps
        #return abs_matrix
        z = np.diagonal(abs_matrix,axis1=1,axis2=2)+eps
        y = z[:, None, :]*z[:, :, None]
        return  np.sum(np.log(np.divide(abs_matrix,np.sqrt(y))),axis=0)

import time
start= time.time()

from joblib import Parallel, delayed 
import multiprocessing


num_cores = multiprocessing.cpu_count()
#results = [ calculateDistance(5000,i) for i in range(0,2)]
results = Parallel(n_jobs=5)(delayed(calculateDistance)(6000,k) for k in range(0,5))

print time.time()- start


