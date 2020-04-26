# Project: MNIST Polynomial Regression Classifier
# Description: Classifies hand-written digits from the MNIST dataset.
# Author: Johnathan Chivington

import os,sys,mnist,time
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
#  Utility Functions
#-----------------------------------------------------------------
def underline(m=''):
    if m == '':
        print('\n No message passed.'); return
    else:
        print(m);[sys.stdout.write((' ' if m[i]==' ' else '-')if i==0 else('\n'if i==len(m) else '-'))for i in range(len(m))]

def clear(m=''):
    os.system('clear'if os.name=='posix'else'cls')
    if m != '':
        print(m)

def greet():
    clear(f'\n MNIST Hand-Written Digit Classifier (Polynomial Regression)')
    underline(f' Johnathan Chivington - john@chivington.io                 ')

def load_dataset():
    print(f'\n Loading MNIST dataset...\n')
    data = mnist.MNIST('./data/')
    Xtrain, Ytrain = map(np.array, data.load_training())
    X_test, Y_test = map(np.array, data.load_testing())
    Xtrain = Xtrain/255.0
    X_test = X_test/255.0
    return Xtrain,Ytrain,X_test,Y_test

def split_dataset(X,Y,p):
    print(f'\n Splitting dataset...')
    n,d = X.shape
    Xtrn,Ytrn = X[np.arange(np.int(n*p))],Y[np.arange(np.int(n*p))]
    Xval,Yval = X[np.arange(np.int(n*p),n-1)],Y[np.arange(np.int(n*p),n-1)]
    return Xtrn,Ytrn,Xval,Yval

def display_digit(flat_digit,label):
    print('\n Displaying image. Close to continue...')
    plt.figure(figsize=plt.figaspect(1.0))
    plt.subplot(1,1,1)
    plt.imshow(flat_digit.reshape([28,28]), cmap=plt.cm.gray)
    plt.title(f'Random MNIST Digit: {label}', fontsize=20)
    plt.show()

def permute(X):
    n,d = X.shape
    idxs = np.arange(n)
    ridx = np.random.permutation(idxs)
    return X[ridx], ridx

def initialize_weights(p,d):
    print(f' Initializing weights...')
    G = np.random.normal(loc=0,scale=np.sqrt(0.1),size=[p,d])
    b = np.random.uniform(0,(np.pi*2),p).reshape([p,1])
    return G,b

def transform_input(X,W,b):
    return np.cos(W.dot(X.T)+b).T

def plot_errors(p,trn,tst,tms):
    plt.figure(figsize=plt.figaspect(0.5))
    plt.style.use('seaborn-whitegrid')
    plt.title(f'Errors w.r.t. Increasing P',fontsize=16,fontweight='bold')
    plt.xlabel(f'P',fontsize=12,fontweight='bold')
    plt.ylabel(f'Errors',fontsize=12,fontweight='bold')
    plt.plot(p,trn.T,'-',color='#58e',label='Training Error')
    plt.plot(p,tst.T,'-',color='#9f9',label='Validation Error')
    plt.plot(p,tms.T,'-',color='#f99',label='Training Times (s)')
    plt.legend(loc='upper right',frameon=True,borderpad=1,borderaxespad=1,facecolor='#fff',edgecolor='#777',shadow=True)
    plt.show()

def hoeffding(X,delta):
    min,max = np.amin(np.sum(X,axis=1)),np.amax(np.sum(X,axis=1))
    return np.sqrt(np.square(max-min)*np.log(2/delta)/(2*X.size))

#-----------------------------------------------------------------
#  Class MNIST_Polynomial_Regression_Classifier
#-----------------------------------------------------------------
class MNIST_Polynomial_Regression_Classifier:
    def __init__(self,lambd=1e-4):
        self.lambd = lambd
        self.k = 10
        self.W = None

    def encode(self,Y):
        n = Y.shape[0]
        one_hot = np.zeros((n,self.k))
        for i in range(n):
            one_hot[i,Y[i]] = 1
        return one_hot

    def train(self,X,Y):
        print(f'\n Training model...')
        reg = self.lambd*np.eye(X.shape[1])
        self.W = np.linalg.pinv(X.T.dot(X)+reg).dot(X.T).dot(self.encode(Y))
        return self.W

    def predict(self,W,X):
        return np.argmax(W.T.dot(X.T),axis=0)

    def err(self,predictions,labels):
        return np.round((1-(np.sum([predictions==labels])/labels.size))*100,10)


#-----------------------------------------------------------------
#  Main / Driver
#-----------------------------------------------------------------
if __name__ == '__main__':
    # Greet user
    greet()

    # Seed RNG
    np.random.seed(4)

    # Define hyperparameters
    lambd = 1e-5
    p = np.linspace(5,1000,10,dtype='int')

    # Error placeholders
    trnErrs,valErrs,times = np.zeros([1,p.size]),np.zeros([1,p.size]),np.zeros([1,p.size])

    # Load train & test data
    Xtrain,Ytrain,Xtest,Ytest = load_dataset()

    # Display sample image
    sample = np.random.randint(0,Xtrain.shape[0])
    display_digit(Xtrain[sample],Ytrain[sample])

    # Instantiate & train model
    print(f' Instantiating model...')
    model = MNIST_Polynomial_Regression_Classifier(lambd)

    # Experiment with p values
    for i in range(p.size):
        print(f'\n\n ----- TESTING P-VAL: {p[i]} -----')

        # Generate W,b & apply transformation
        G,b = initialize_weights(p[i],Xtrain.shape[1])
        Xtrx = transform_input(Xtrain,G,b)

        # Split dataset
        Xtrn,Ytrn,Xval,Yval = split_dataset(Xtrx,Ytrain,0.8)

        # Train model & record training time
        begin = time.time()
        Wp = model.train(Xtrx,Ytrain)
        train_time = np.round(time.time()-begin,3)
        print(f' Training Time: {train_time}s')

        #  Make predictions
        print('\n Making predictions...')
        Ytrn_hat,Yval_hat = model.predict(Wp,Xtrn),model.predict(Wp,Xval)

        # Calculate & display error
        print(f'\n Calculating error...')
        trnErr,valErr = model.err(Ytrn_hat,Ytrn),model.err(Yval_hat,Yval)
        trnErrs[0,i]=trnErr; valErrs[0,i]=valErr; times[0,i]=train_time
        print(f' Train Subset Error: {trnErr}\n Validation Error: {valErr}\n\n')

        if (i==p.size-1):
            # Train on full Xtrain & plot
            print(f'\n\n ----- FINAL P-VAL: {p[p.size-1]} -----')
            Ytrain_hat,Ytest_hat = model.predict(Wp,Xtrx),model.predict(Wp,transform_input(Xtest,G,b))
            trainErr,testErr = model.err(Ytrain_hat,Ytrain),model.err(Ytest_hat,Ytest)
            trainInt,testInt = np.round(hoeffding(Xtrain,0.05),4),np.round(hoeffding(Xtest,0.05),4)
            print(f' Final Train Error: {trainErr} +/- {trainInt}\n Final Test Error: {testErr} +/- {testInt}\n\n')
            plot_errors(p,trnErrs,valErrs,times)
