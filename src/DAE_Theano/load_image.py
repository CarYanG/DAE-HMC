import numpy as np
import theano.tensor as T
import theano
import scipy.misc as misc
import os
from matplotlib.tests.test_rcparams import fname
from theano.scalar.basic import int32
# import Image
class data_reader():
    def __init__(self,path,label=None,batch_size=100,patch=256):
        self.dir_path = path
        self.batch_size = batch_size
        self.file_list = list()
        for dirname,_,filename in os.walk(self.dir_path):
            for fname in filename:
                self.file_list.append(os.path.join(dirname,fname))
        self._batch_str = 0
        if label is not None:
            self._label = self.load_label(label)
        else:
            self._label = None
        self.patch = patch;


    def reinit(self):
        self._batch_str = 0

    def normalize(self,img):
        return (img - np.min(img))/(np.max(img)-np.min(img))

    def _to_gray(self,img):
        img = np.sum(img,axis=2)
        return np.array(img,dtype=np.float32)

    def _get_patches(self,img):
        sz = img.itemsize
        h,w = img.shape
        bh,bw=self.patch,self.patch
        shape=(h/bh,w/bh,bh,bw)
        strides = sz*np.array([w*bh,bw,w,1])
        blocks=np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
        return np.reshape(blocks,(-1,self.patch*self.patch))

    def next_batch(self):
        data_list = list()
        for fp in self.file_list[self._batch_str:self._batch_str+self.batch_size]:
#        print fp
            img = misc.imread(fp)
            img = self._to_gray(img)
            img = self.normalize(img)
            patch = self._get_patches(img)
            data_list.append(patch)
        data_list = np.array(data_list,dtype=np.float32)

        data_list = np.reshape(data_list,(-1,self.patch*self.patch))
        shared_list=theano.shared(np.asarray(data_list,
                                               dtype=theano.config.floatX),borrow=True)

        if self._label is not None:
            label_temp = self._label[self._batch_str:self._batch_str+self.batch_size,:]
            label_temp[label_temp<0] = 0

        self._batch_str = self._batch_str +self.batch_size

        if self._label is not None:
            label_temp = theano.shared(np.asarray(label_temp,dtype=theano.config.floatX),borrow=True)
            return shared_list,T.cast(label_temp, 'int32')

        else:
            return shared_list

    def load_label(self,path):

        label = list()
        for dirname,_,filename in os.walk(path):
            for fname in filename:
                temp = list()
                with open(os.path.join(dirname,fname)) as f:
                    for line in f:
                        val = line.split(" ")
                        temp.append(int(val[len(val)-1]))

                label.append(np.array(temp))
        return np.transpose(np.array(label))