import os
import numpy as np
from PIL import Image
from numpy import linalg
import math
import matplotlib.pyplot as plt

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    import scipy.linalg as linalg
    import operator
    ew, ev = linalg.eig(B,W+B)
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.transpose(np.array(Centers))
    return LDAW,Centers, classLabels

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1)
    A = A - np.transpose(np.tile(m, (c,1)))
    B = np.dot(np.transpose(A), A)
    [d,v] = linalg.eig(B)
    # v is in descending sorted order

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m


def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        infile = os.path.join(directory, f)
        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A)
    faces = np.transpose(faces)
    idLabel = np.array(Label)

    return faces,idLabel

#input data
os.chdir("C:/Users/qazsweet/Documents/python/TianMengge_assignment5")
facestest, idLabeltest = read_faces('C:/Users/qazsweet/Documents/python/TianMengge_assignment5/test')
facestrain, idLabeltrain = read_faces('C:/Users/qazsweet/Documents/python/TianMengge_assignment5/train')

def Accuracy(a):
    K = 30
    [W_pca,LL,m_pca]=myPCA(facestest)
    We = W_pca[:,: K]
    xm = np.zeros((22400,120))
    for i in range(0,120):
        xm[:,i] =facestest[:,i]-m_pca
    y = np.dot(np.transpose(We), xm)
    Z = np.zeros((30,10))
    m =0
    for i in range(0,30):
        for n in range(0,120):
            while idLabeltest[n] == m:
                Z[:,m] = Z[:,m] + y[:,n]
                m = m+1
                if m == 10:
                    break
    Z = Z/12  

    K1 = 90
    W1 = W_pca[:,: K1]
    x1 = np.zeros((90,120))
    for i in range(0,120):
        x1[:,i] = np.dot(np.transpose(W1), (facestest[:,i]-m_pca))
    r=ComputeNorm(facestest)
    LDAWt,Centerst, classLabelst = myLDA(x1,idLabeltest)
    yy = np.dot(np.transpose(LDAWt), np.transpose(W1))
    yf = np.zeros((9,120))
    yf = np.dot(yy, xm)

    ytest = np.vstack([a*y,(1-a)*yf])
    ytemplate = np.vstack([a*Z,(1-a)*Centerst])

    K = 30
    [W_pcat,LLt,m_pcat]=myPCA(facestrain)
    Wet = W_pcat[:,: K]
    xmt = np.zeros((22400,120))
    for i in range(0,120):
        xmt[:,i] =facestrain[:,i]-m_pca
    yt = np.dot(np.transpose(Wet), xmt)
    Zt = np.zeros((30,10))
    mt =0
    for i in range(0,30):
        for n in range(0,120):
            while idLabeltrain[n] == mt:
                Zt[:,mt] = Zt[:,mt] + yt[:,n]
                mt = mt+1
                if mt == 10:
                    break
    Zt = Zt/12  

    K1 = 90
    W1t = W_pcat[:,: K1]
    x1t = np.zeros((90,120))
    for i in range(0,120):
        x1t[:,i] = np.dot(np.transpose(W1t), (facestrain[:,i]-m_pcat))
    r=ComputeNorm(facestrain)
    LDAWt,Centerstt, classLabelst = myLDA(x1,idLabeltrain)
    yyt = np.dot(np.transpose(LDAWt), np.transpose(W1t))
    yft = np.zeros((9,120))
    yft = np.dot(yyt, xmt)

#feature fusion
    ytrain = np.vstack([a*yt,(1-a)*yft])
    ytemplatetrain = np.vstack([a*Zt,(1-a)*Centerstt])

    Labeltest = []
    for x in range(0,10):
        com = np.zeros((39,10))
        for i in range(0,10):
            com[:,i] = ytemplatetrain[:,i] - ytemplate[:,x]
        r8=ComputeNorm(com)
        r8 = r8.argmin(axis=1)
        Labeltest.append(int(r8))
    Acc = []
    su = 0
    for i in range(0,10):
        if classLabelst[i] == Labeltest[i]:
            su+=1
    su = np.float32(su/10)
    Acc.append(su)
    print('a = ',a)
    #print(Labeltest)
    #print(Acc)
    print(Zt.shape)
    return Acc

for ss in range(0,11):
    ss = ss/10
    Acc = Accuracy(ss)
    plt.scatter(ss,Acc)

#plt.savefig("PCA &LDA.png")
plt.show()
