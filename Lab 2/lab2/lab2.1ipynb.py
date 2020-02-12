#!/usr/bin/env python
# coding: utf-8

# In[128]:


import numpy as np
import matplotlib.pyplot as plt
import random 
import os

class RBFNetwork():
    
    def __init__(self, version="sin",std=None,rbf_units=4,eta=0.1, threshold=0.1, plot_rbf_pred=False, place_ver=0,CL_eta=0.1,use_seed=True,rd_seed=10):
        self.use_seed=use_seed
        #if self.use_seed:
        #    np.random.seed(401)
        self.x_train = np.array([np.arange(0, 2*np.pi, 0.1)])
        self.x_test  = np.array([np.arange(0.05, 2*np.pi, 0.1)])
        self.x_train = np.arange(0, 2*np.pi, 0.1)
        self.x_test  = np.arange(0.05, 2*np.pi, 0.1)
        self.y_train = np.sin(2*self.x_train)
        self.y_test  = np.sin(2*self.x_test)        
        if version=="square":            
            self.y_train = np.where(np.sign(self.y_train)>=0, 1.0, -1.0,)
            self.y_test  = np.where(np.sign(self.y_test)>=0, 1.0, -1.0)
        self.rbf_units=rbf_units
        np.random.seed(rd_seed)
        self.rbf=[np.arange(2*np.pi/(2*rbf_units), 2*np.pi, 2*np.pi/rbf_units),np.arange(0, 2*np.pi, 2*np.pi/rbf_units),
                                                          np.random.rand(rbf_units)*(2*np.pi),self.place_func(rbf_units)][place_ver]    
        
        self.CL_std=False
        if std==None and rbf_units!=1:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        elif std==None and rbf_units==1:
            self.std=1/np.sqrt(2*rbf_units)
        
        else: 
            self.std=std
        self.W=np.random.rand(rbf_units)
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        self.std_list=None
        self.cl_std=False
        
        self.plot_rbf_pred=plot_rbf_pred
        self.plot_rbf_pred_ballist=False
        if self.plot_rbf_pred:
            plt.figure(0)
            plt.plot(self.rbf,np.ones_like(self.rbf)*rbf_units,"*")
            plt.figure(4*rbf_units)
            plt.plot(self.x_train,self.y_train,"+")
            plt.plot(self.rbf,np.zeros_like(self.rbf),"*")
            
    def place_func(self,nr_units):
        place_list=np.arange(np.pi/4.0, 2*np.pi, np.pi/2.0)[0:nr_units]
        if nr_units>4:
            list_rbf=[2*np.pi]
            list_rbf2=np.arange(0, 2*np.pi, 2*np.pi/(max(nr_units-5,1.0)))
            for i in place_list: list_rbf+=[i]
            
            if nr_units>5:
                for i in list_rbf2:  list_rbf+=[i]
            place_list=np.array(list_rbf)
        return place_list

    def plot_data(self):
        plt.plot(self.x_train,self.y_train,"*")
        plt.plot(self.x_test,self.y_test,"*")

    def add_noise(self, test=True):
        if self.use_seed:
            np.random.seed(401)
        self.x_train +=np.random.normal(0,0.1,np.shape(self.x_train))
        self.y_train +=np.random.normal(0,0.1,np.shape(self.x_train))
        if test:
            self.x_test +=np.random.normal(0,0.1,np.shape(self.x_train))
            self.y_test +=np.random.normal(0,0.1,np.shape(self.x_train))
        return self.x_train , self.x_test, self.y_train, self.y_test
        

    def calcRbf(self,x,r,s,option=0):
        if not self.cl_std:
            return np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2)))
        else:
            print("jasda",np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2))))
            print("\n\nalksd",np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(self.std_list,2))))
            
            print("\n\n\n\n\n\n\n\n", np.shape(np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2)))))
            print("\n\n\n\n\n\n\n\n", np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(self.std_list,2))))
        
    
    def forward(self, X, transform, option=0): 
        phi=self.calcRbf(X,self.rbf,self.std, option)
        y_pred=np.dot(phi,self.W)
        if transform:
                y_pred = np.sign(y_pred)
        return y_pred , phi
          
    def batch_mode_training(self,transform):
        phi=self.calcRbf(self.x_train,self.rbf,self.std)
        self.W = np.dot(np.linalg.pinv(phi),self.y_train)
        y_pred=np.dot(phi,self.W)
        if transform:
            y_pred = np.sign(y_pred)        
        error=np.mean(abs(self.y_train-y_pred))        
        return error
            
    def on_line_learning(self,transform):
        # one epoch
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        error_list=[]
        for i in range(len(X)):
            x=X[i]
            y=Y[i]
            y_pred, phi=self.forward(x, transform, option=1)
            e=y-y_pred
            error_list+=[np.sqrt(np.dot(e,e))]
            if np.size(e)==1:
                delta_W=self.eta*e*phi
            else:
                delta_W=np.array([self.eta*e[i]*phi for i in range(np.size(e))]).T
            self.W += delta_W
        #error=np.sqrt(np.dot(e,e))
        for i in range(len(X)):
            x=X[i]
            y=Y[i]
            y_pred, phi=self.forward(x, transform, option=1)
            e=y-y_pred
            error_list+=[np.sqrt(np.dot(e,e))]
        error=np.mean(error_list)
        
        
        return error
                    
    def run(self, threshold=None,transform=False, mode="batch"):
        self.mode=mode
        if threshold==None:
            threshold=self.threshold
            
        if self.mode=="batch":
            error_train=self.batch_mode_training(transform)
            y_pred,_=self.forward(self.x_test, transform)  
            error_test=np.mean(abs(self.y_test-y_pred))  
            
        else:
            error=np.inf
            error_train=99999999999
            error_test=99999999999
            count=0
            error_list1=[]
            error_list2=[]
            conditional_statement=True
            while (len(np.transpose(self.rbf))*10000)>count:# and error_test>self.threshold and conditional_statement:#  (error-error_test)>1e-180 and count < 999999 and error>error_test:
                count+=1
                error_train=self.on_line_learning(transform)
                y_pred,_=self.forward(self.x_test, transform)
                error_test=np.mean(abs(self.y_test-y_pred))
                error_list1+=[error_train]
                error_list2+=[error_test]
                
            if self.plot_rbf_pred:
                plt.figure(4*self.rbf_units+2)
                plt.plot(range(count),error_list1,"b")
                plt.plot(range(count),error_list2,"r")
            
            elif self.plot_rbf_pred_ballist:
                plt.figure(4*self.rbf_units+2)
                plt.plot(range(count),error_list1,"b")
                plt.plot(range(count),error_list2,"r")
                
          
        if self.plot_rbf_pred:
            plt.figure(4*self.rbf_units+3)
            plt.plot(self.rbf,np.zeros_like(self.rbf),"+")
            plt.plot(self.x_test,y_pred,"-")
        return error_train , error_test
    
    def run_CL(self, threshold=None,transform=False, mode="batch",nr_winners=1):
        self.CL(nr_winners)   
        return self.run(threshold,transform,mode)
                        
    def CL(self,nr_winners):
        
        for j in range(10000):#(7*500):
            ind_list = [i for i in range(np.size(self.x_train))]
            random.shuffle(ind_list)
            X = self.x_train [ind_list]
            Y = self.y_train [ind_list]
            #for i in range(1):#len(X)):
            Winner=self.winner(X[0],self.rbf,nr_winners)
            for k in Winner:
                self.rbf[k]+=self.CL_eta*(X[0]-self.rbf[k])
                    
        if self.CL_std:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*self.rbf_units)
            
        
        if self.plot_rbf_pred:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.rbf,np.ones_like(self.rbf)*np.size(self.rbf),"*")
        
        
        distance_rx_list=[]
        for i in range(len(self.x_train)):
            #print("ää",self.winner(self.x_train[i],self.rbf,nr_winners=1))
            #print(type(self.winner(self.x_train[i],self.rbf,nr_winners=1)))
            distance_rx_list+=list(self.winner(self.x_train[i],self.rbf,nr_winners=1))
        distance_rx_list=np.array(distance_rx_list)
        #print("distance_rx_list",distance_rx_list)
        std_list=[]
        for i in range(self.rbf_units):
            std_list+=[np.sqrt(np.mean(np.power(self.x_train[np.where(np.array(distance_rx_list)==i)]-self.rbf[i],2)))]
        self.std_list=np.array(std_list)
        self.cl_std=True
        print(std_list)
                
            
        
            
    def winner(self,x,rbf, nr_winners=1):
        distance_list=[[abs(x-rbf[i]),i] for i in range(len(rbf))]
        distance_list.sort()
        winner_list=np.array(distance_list,dtype=int)[:nr_winners][:,1]
        return winner_list


class RBFNetwork_ballist(RBFNetwork):
    
    def __init__(self, version="sin",std=None,rbf_units=5,eta=0.1, threshold=0.1, plot_rbf_pred_ballist=False, place_ver=0,
                 CL_eta=0.1, CL_plot=True):
        with open(os.path.abspath("ballist.dat")) as file: train_data=np.array([data.split() for data in file],dtype=float)
        with open(os.path.abspath("balltest.dat")) as file: test_data=np.array([data.split() for data in file], dtype=float)
        self.x_train = train_data[:,:2]
        self.x_test  = test_data[:,:2]
        self.y_train = train_data[:,2:]
        self.y_test  = test_data[:,2:]
        
        self.rbf=[np.arange(1/(2*rbf_units), 1, 1/rbf_units),np.arange(0, 1, 1/rbf_units), np.random.rand(rbf_units),
                                                                                  self.place_func(rbf_units)][place_ver]
        self.rbf=np.random.rand(2,rbf_units)#[self.rbf,self.rbf]
        self.CL_std=False
        if std==None and rbf_units==1: self.std=1.0/np.sqrt(2)
        elif std==None: 
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        else:self.std=std
        self.rbf_units=rbf_units
        self.W=np.random.rand(rbf_units,2)
        
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        self.plot_rbf_pred_ballist=plot_rbf_pred_ballist
        self.plot_rbf_pred=False
        self.CL_plot=CL_plot
        if self.plot_rbf_pred_ballist:
            plt.figure(0)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
        if self.plot_rbf_pred_ballist: 
            plt.figure(4*rbf_units)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")
            
    def calcRbf(self,x,r,s,option=0):
        if not option==1:
            return np.exp(-np.power(sum([np.transpose(np.array([x[:,i]]))-r[i] for i in range(np.shape(x)[1])]),2)/(2.0*np.power(s,2)))  
        else:
            return np.exp(-np.power(sum([np.transpose(x[i])-r[i] for i in range(int(np.size(x)))]),2)/(2.0*np.power(s,2)))  

        
    def run(self, threshold=None,transform=False, mode="batch"):
        error_train , error_test=super().run(threshold,transform, mode)
        if self.plot_rbf_pred_ballist:
            y_pred,_=self.forward(self.x_test, transform)
            plt.figure(4*self.rbf_units+3)
            plt.plot(self.y_test[:,0],self.y_test[:,1],".")
            plt.plot(y_pred[:,0],y_pred[:,1],"*")
        return error_train , error_test
    
    def winner(self,x,rbf):
        return np.argmin(np.sqrt(np.power(x[0]-rbf[0],2)+np.power(x[1]-rbf[1],2)))

    def CL(self,nr_winners):
        #add more winners
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        for j in range(10000):#(7*500):
            #for i in range(len(X)):
            Winner=self.winner(X[0],self.rbf)
            self.rbf[:,Winner]+=self.CL_eta*(X[0]-self.rbf[:,Winner])
        if self.CL_std:
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*self.rbf_units)
            
        if self.CL_plot:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")


# In[30]:


import numpy as np


class SLP:
    """Single-Layer Perceptron (neural network with one layer of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, method, mode, bias=True, animation=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.weights = None
        self.mce = None                 # misclassification error
        self.animation = animation      # plot decision boundary at each epoch

        # select method
        if method == 'delta':
            self.delta = True
        elif method == 'perceptron':
            self.delta = False
        else:
            exit(-1)

        # select mode
        if mode == 'batch':
            self.batch = True
        elif mode == 'sequential':
            self.batch = False
        else:
            exit(-1)

    def learn(self, patterns, targets):
        """Train the perceptron using the Delta learning rule."""
        if self.bias:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)  # add row for bias term
        self.weights = self._sigma * np.random.randn(targets.shape[0], patterns.shape[0])   # init weights

        # stats
        y = self.predict(patterns)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        for i in range(self.epochs):
            # animation of decision boundary
            if self.animation:
                xlim = [min(patterns[0]) - 0.1, max(patterns[0]) + 0.1, min(patterns[1]) - 0.1, max(patterns[1]) + 0.1]
                self.plot_decision_boundary(xlim, 'y')

            # update weights
            if self.batch:
                if self.delta:
                    e = targets - self.weights @ patterns
                else:
                    e = targets - self.predict(patterns)
                self.weights += self.learning_rate * e @ patterns.T
            else:
                for n in range(patterns.shape[1]):
                    if self.delta:
                        e = targets[:, n] - self.weights @ patterns[:, n]
                    else:
                        e = targets[:, n] - self.predict(patterns[:, n])
                    self.weights += self.learning_rate * e @ patterns[:, n].reshape(1, -1)

            # stats
            y = self.predict(patterns)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

    def predict(self, patterns):
        if patterns.shape[0] != self.weights.shape[1]:  # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        return self._activate(self.weights @ patterns)

    def _activate(self, x):
        if self.delta:
            targets = (1, -1)
        else:
            targets = (1, 0)
        return np.where(x > 0, targets[0], targets[1])

    _sigma = 0.01   # standard deviation for weight initialization


class MLP:
    """Two-layer perceptron (neural network with two layers of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, hidden_nodes, alpha=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_nodes = hidden_nodes    # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer
        self.mce = None                     # misclassification error

    def learn(self, patterns, targets):
        """Train the perceptron using BackProp with momentum."""
        # init (see Marsland's book for initialization of weights)
        patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        self.W = 1. / np.sqrt(patterns.shape[0]) * (np.random.rand(self.hidden_nodes, patterns.shape[0]) - 0.5)
        self.V = 1. / np.sqrt(self.hidden_nodes+1) * (np.random.rand(targets.shape[0], self.hidden_nodes+1) - 0.5)
        dW = 0
        dV = 0

        # stats
        y = self.predict(patterns)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        for i in range(self.epochs):
            # forward pass
            H, O = self._forward(patterns)

            # backward pass
            delta_o = (O - targets) * self._derivative_activate(O)
            delta_h = (self.V.T @ delta_o) * self._derivative_activate(H)
            delta_h = delta_h[:-1]                                                      # remove extra row

            # weight update
            dW = self.alpha * dW - (1 - self.alpha) * (delta_h @ patterns.T)
            dV = self.alpha * dV - (1-self.alpha) * (delta_o @ H.T)
            self.W += self.learning_rate * dW
            self.V += self.learning_rate * dV

            # stats
            y = self.predict(patterns)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

    def predict(self, patterns):
        if patterns.shape[0] != self.W.shape[1]:    # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        O = self._forward(patterns)[1]
        O = np.where(abs(O) < 1e-6, 0, O)           # on the decision boundary
        O = np.where(O > 0, 1, O)
        O = np.where(O < 0, -1, O)
        return O

    def _forward(self, patterns):
        H = self._activate(self.W @ patterns)
        H = np.concatenate((H, np.ones((1, patterns.shape[1]))), axis=0)                # add row for bias term
        O = self._activate(self.V @ H)
        return H, O

    def _activate(self, x):
        return 2. / (1 + np.exp(-x)) - 1

    def _derivative_activate(self, phi_x):
        return (1 + phi_x) * (1 - phi_x) / 2


# In[129]:


he=RBFNetwork(rbf_units=4, plot_rbf_pred=True, place_ver=2)
he.run_CL(nr_winners=1)


# In[50]:


#3.3.2
he=RBFNetwork(rbf_units=81, plot_rbf_pred=True, place_ver=2)
he.run_CL(nr_winners=10)

#rbf_NN=RBFNetwork(rbf_units=1, std=None, version="sin", place_ver=0)#, plot_rbf_pred=True)
#error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
#error_list1+=[error_train]
#error_list2+=[error_test]
#print(error_test)



std_=None
count=0
error_train=np.inf
error_test=np.inf
error_list1=[]
error_list2=[]
for threshold in [0.1]:#,0.01, 0.001, 0]:
    while error_test> threshold:
        count+=1
        rbf_NN=RBFNetwork_ballist(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, place_ver=0, 
                                 plot_rbf_pred_ballist=True)
        rbf_NN.add_noise(test=False)
        error_train , error_test=rbf_NN.run_CL()#mode="online")
        error_list1+=[error_train]
        error_list2+=[error_test]
        if count>5:
            break
#plt.figure(0)
#plt.plot(range(1,count+1),error_list1,"b")
#plt.plot(range(1,count+1),error_list2,"r")
print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
      "\nnr of units",count,"\n-------------")
print("\n")


# In[24]:


#3.3.3
std_=None
count=0
error_train=np.inf
error_test=np.inf
error_list1=[]
error_list2=[]
for threshold in [0.01]:#,0.01, 0.001, 0]:
    while error_test> threshold:
        count+=1
        rbf_NN=RBFNetwork_ballist(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, place_ver=0, 
                                 plot_rbf_pred_ballist=True)
        rbf_NN.add_noise(test=False)
        error_train , error_test=rbf_NN.run_CL()#mode="online")
        error_list1+=[error_train]
        error_list2+=[error_test]
        if count>2:
            break
#plt.figure(0)
#plt.plot(range(1,count+1),error_list1,"b")
#plt.plot(range(1,count+1),error_list2,"r")
print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
      "\nnr of units",count,"\n-------------")
print("\n")
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


#3.2 amout of units per threshold not correct 
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None
    count=4
    error_train=np.inf
    error_test=np.inf
    for threshold in [0.1,0.01, 0.001, 0]:
        count-=1
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            #rbf_NN.add_noise(test=False)
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            #error_train , error_test=rbf_NN.run(transform=transform_)
            if count>80:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[4]:


#3.2
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None
    count=4
    error_train=np.inf
    error_test=np.inf
    for threshold in [0.01]:#,0.01, 0.001, 0]:
        count-=1
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=6,threshold=threshold ,eta=0.01, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            #rbf_NN.add_noise(test=False)
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            #error_train , error_test=rbf_NN.run(transform=transform_)
            if count>1:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[10]:


for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None
    count=4
    error_train=np.inf
    error_test=np.inf
    for threshold in [0.01]:#,0.01, 0.001, 0]:
        count-=1
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=6,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            #rbf_NN.add_noise(test=False)
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            #error_train , error_test=rbf_NN.run(transform=transform_)
            if count>1:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[9]:


#3.2

for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    
    for threshold in [0.01]:#, 0.01, 0.001, 0]:
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=count, std=std_, version=func, place_ver=0, plot_rbf_pred=True)
            #rbf_NN.add_noise(test=False)
            #rbf_NN.add_noise()
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            #error_list1+=[error_train]
            #error_list2+=[error_test]
            if count>100:
                break
            print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
                  "\nnr of units",count,"\n-------------")
        #plt.figure(i)
        #plt.plot(range(1,count+1),error_list1,"b")
        #plt.plot(range(1,count+1),error_list2,"r")
    print("\n")


    


# In[ ]:





# In[ ]:


#3.3
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None#
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        #elif transform_==True:
        threshold=0.0001
        #while error_test> threshold:
        #count+=1
        rbf_NN=RBFNetwork(rbf_units=10, threshold=threshold,CL_eta=0.01, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
        #rbf_NN.add_noise(test=False)
        error_train , error_test=rbf_NN.run_CL(mode="online",transform=transform_)
        if count>200:
            break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[2]:


# 3.1

#sin(2x)
#training set & test set | only training set
# std=1                  | std = d/sqrt(2M)
# 0.1 -> 5    (6)        | 0.1 -> 52
# 0.01 -> 10  (10)       | 0.01 -> 61
# 0.001 -> 13 (12)       | 0.001 -> 63
#_______

# * error 0 for square(2x) using the transformed output -> 7 units \ std=1

for i in range(3):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=#None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    for threshold in [0.1, 0.01, 0.001, 0]:
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=count, std=std_, version=func, place_ver=0, plot_rbf_pred=True)
            error_train , error_test=rbf_NN.run(transform=transform_)
            if count>20:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[111]:


import numpy as np


class SLP:
    """Single-Layer Perceptron (neural network with one layer of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, method, mode, bias=True, animation=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.weights = None
        self.mce = None                 # misclassification error
        self.animation = animation      # plot decision boundary at each epoch

        # select method
        if method == 'delta':
            self.delta = True
        elif method == 'perceptron':
            self.delta = False
        else:
            exit(-1)

        # select mode
        if mode == 'batch':
            self.batch = True
        elif mode == 'sequential':
            self.batch = False
        else:
            exit(-1)

    def learn(self, patterns, targets):
        """Train the perceptron using the Delta learning rule."""
        if self.bias:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)  # add row for bias term
        self.weights = self._sigma * np.random.randn(targets.shape[0], patterns.shape[0])   # init weights

        # stats
        y = self.predict(patterns)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        for i in range(self.epochs):
            # animation of decision boundary
            if self.animation:
                xlim = [min(patterns[0]) - 0.1, max(patterns[0]) + 0.1, min(patterns[1]) - 0.1, max(patterns[1]) + 0.1]
                self.plot_decision_boundary(xlim, 'y')

            # update weights
            if self.batch:
                if self.delta:
                    e = targets - self.weights @ patterns
                else:
                    e = targets - self.predict(patterns)
                self.weights += self.learning_rate * e @ patterns.T
            else:
                for n in range(patterns.shape[1]):
                    if self.delta:
                        e = targets[:, n] - self.weights @ patterns[:, n]
                    else:
                        e = targets[:, n] - self.predict(patterns[:, n])
                    self.weights += self.learning_rate * e @ patterns[:, n].reshape(1, -1)

            # stats
            y = self.predict(patterns)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

    def predict(self, patterns):
        if patterns.shape[0] != self.weights.shape[1]:  # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        return self._activate(self.weights @ patterns)

    def _activate(self, x):
        if self.delta:
            targets = (1, -1)
        else:
            targets = (1, 0)
        return np.where(x > 0, targets[0], targets[1])

    _sigma = 0.01   # standard deviation for weight initialization


class MLP:
    """Two-layer perceptron (neural network with two layers of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, hidden_nodes, alpha=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_nodes = hidden_nodes    # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer
        self.mce = None                     # misclassification error

    def learn(self, patterns, targets):
        """Train the perceptron using BackProp with momentum."""
        # init (see Marsland's book for initialization of weights)
        patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        self.W = 1. / np.sqrt(patterns.shape[0]) * (np.random.rand(self.hidden_nodes, patterns.shape[0]) - 0.5)
        self.V = 1. / np.sqrt(self.hidden_nodes+1) * (np.random.rand(targets.shape[0], self.hidden_nodes+1) - 0.5)
        dW = 0
        dV = 0

        # stats
        y = self.predict(patterns)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        for i in range(self.epochs):
            # forward pass
            H, O = self._forward(patterns)

            # backward pass
            delta_o = (O - targets) * self._derivative_activate(O)
            delta_h = (self.V.T @ delta_o) * self._derivative_activate(H)
            delta_h = delta_h[:-1]                                                      # remove extra row

            # weight update
            dW = self.alpha * dW - (1 - self.alpha) * (delta_h @ patterns.T)
            dV = self.alpha * dV - (1-self.alpha) * (delta_o @ H.T)
            self.W += self.learning_rate * dW
            self.V += self.learning_rate * dV

            # stats
            y = self.predict(patterns)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

    def predict(self, patterns):
        if patterns.shape[0] != self.W.shape[1]:    # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        O = self._forward(patterns)[1]
        #O = np.where(abs(O) < 1e-6, 0, O)           # on the decision boundary
        #O = np.where(O > 0, 1, O)
        #O = np.where(O < 0, -1, O)
        return O

    def _forward(self, patterns):
        H = self._activate(self.W @ patterns)
        H = np.concatenate((H, np.ones((1, patterns.shape[1]))), axis=0)                # add row for bias term
        O = self._activate(self.V @ H)
        return H, O

    def _activate(self, x):
        return 2. / (1 + np.exp(-x)) - 1

    def _derivative_activate(self, phi_x):
        return (1 + phi_x) * (1 - phi_x) / 2


# In[138]:


import numpy as np
import matplotlib.pyplot as plt
import random 
import os

class RBFNetwork():
    
    def __init__(self, version="sin",std=None,rbf_units=4,eta=0.1, threshold=0.1, plot_rbf_pred=False, place_ver=0,CL_eta=0.1,use_seed=True):
        self.use_seed=use_seed
        #if self.use_seed:
        #    np.random.seed(401)
        self.x_train = np.array([np.arange(0, 2*np.pi, 0.1)])
        self.x_test  = np.array([np.arange(0.05, 2*np.pi, 0.1)])
        self.x_train = np.arange(0, 2*np.pi, 0.1)
        self.x_test  = np.arange(0.05, 2*np.pi, 0.1)
        self.y_train = np.sin(2*self.x_train)
        self.y_test  = np.sin(2*self.x_test)        
        if version=="square":            
            self.y_train = np.where(np.sign(self.y_train)>=0, 1.0, -1.0,)
            self.y_test  = np.where(np.sign(self.y_test)>=0, 1.0, -1.0)
        self.rbf_units=rbf_units
        self.rbf=[np.arange(2*np.pi/(2*rbf_units), 2*np.pi, 2*np.pi/rbf_units),np.arange(0, 2*np.pi, 2*np.pi/rbf_units),
                                                          np.random.rand(rbf_units)*(2*np.pi),self.place_func(rbf_units)][place_ver]    
        
        self.CL_std=False
        if std==None and rbf_units!=1:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        elif std==None and rbf_units==1:
            self.std=1/np.sqrt(2*rbf_units)
        
        else: 
            self.std=std
        self.W=np.random.rand(rbf_units)
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        
        self.plot_rbf_pred=plot_rbf_pred
        self.plot_rbf_pred_ballist=False
        #if self.plot_rbf_pred:
        #    plt.figure(0)
        #    plt.plot(self.rbf,np.ones_like(self.rbf)*rbf_units,"*")
        #    plt.figure(4*rbf_units)
        #    plt.plot(self.x_train,self.y_train,"+")
        #    plt.plot(self.rbf,np.zeros_like(self.rbf),"*")
            
    def place_func(self,nr_units):
        place_list=np.arange(np.pi/4.0, 2*np.pi, np.pi/2.0)[0:nr_units]
        if nr_units>4:
            list_rbf=[2*np.pi]
            list_rbf2=np.arange(0, 2*np.pi, 2*np.pi/(max(nr_units-5,1.0)))
            for i in place_list: list_rbf+=[i]
            
            if nr_units>5:
                for i in list_rbf2:  list_rbf+=[i]
            place_list=np.array(list_rbf)
        return place_list

    def plot_data(self):
        plt.plot(self.x_train,self.y_train,"*")
        plt.plot(self.x_test,self.y_test,"*")

    def add_noise(self, test=True):
        if self.use_seed:
            np.random.seed(401)
        plt.figure(512)
        if self.rbf_units==6:
            plt.plot(self.x_test,self.y_test,"*",label="test data_true")
        self.x_train +=np.random.normal(0,0.1,np.shape(self.x_train))
        self.y_train +=np.random.normal(0,0.1,np.shape(self.x_train))
        if test:
            self.x_test +=np.random.normal(0,0.1,np.shape(self.x_train))
            self.y_test +=np.random.normal(0,0.1,np.shape(self.x_train))
        if self.rbf_units==6:
                plt.plot(self.x_test,self.y_test,"+",label="test data_noise")
        return self.x_train , self.x_test, self.y_train, self.y_test
        

    def calcRbf(self,x,r,s,option=0):
        return np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2)))              
    
    def forward(self, X, transform, option=0): 
        phi=self.calcRbf(X,self.rbf,self.std, option)
        y_pred=np.dot(phi,self.W)
        if transform:
                y_pred = np.sign(y_pred)
        return y_pred , phi
          
    def batch_mode_training(self,transform):
        phi=self.calcRbf(self.x_train,self.rbf,self.std)
        self.W = np.dot(np.linalg.pinv(phi),self.y_train)
        y_pred=np.dot(phi,self.W)
        if transform:
            y_pred = np.sign(y_pred)        
        error=np.mean(abs(self.y_train-y_pred))        
        return error
            
    def on_line_learning(self,transform):
        # one epoch
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        error_list=[]
        for i in range(len(X)):
            x=X[i]
            y=Y[i]
            y_pred, phi=self.forward(x, transform, option=1)
            e=y-y_pred
            error_list+=[np.sqrt(np.dot(e,e))]
            if np.size(e)==1:
                delta_W=self.eta*e*phi
            else:
                delta_W=np.array([self.eta*e[i]*phi for i in range(np.size(e))]).T
            self.W += delta_W
        #error=np.sqrt(np.dot(e,e))
        for i in range(len(X)):
            x=X[i]
            y=Y[i]
            y_pred, phi=self.forward(x, transform, option=1)
            e=y-y_pred
            error_list+=[np.sqrt(np.dot(e,e))]
        error=np.mean(error_list)
        
        
        return error
                    
    def run(self, threshold=None,transform=False, mode="batch"):
        self.mode=mode
        if threshold==None:
            threshold=self.threshold
            
        if self.mode=="batch":
            error_train=self.batch_mode_training(transform)
            y_pred,_=self.forward(self.x_test, transform)  
            error_test=np.mean(abs(self.y_test-y_pred))  
            
        else:
            error=np.inf
            error_train=99999999999
            error_test=99999999999
            count=0
            error_list1=[]
            error_list2=[]
            conditional_statement=True
            while (len(np.transpose(self.rbf))*100)>count:# and error_test>self.threshold and conditional_statement:#  (error-error_test)>1e-180 and count < 999999 and error>error_test:
                count+=1
                error_train=self.on_line_learning(transform)
                y_pred,_=self.forward(self.x_test, transform)
                error_test=np.mean(abs(self.y_test-y_pred))
                error_list1+=[error_train]
                error_list2+=[error_test]
                
            if self.plot_rbf_pred:
                plt.figure(4*self.rbf_units+2)
                plt.plot(range(count),error_list1,"b")
                plt.plot(range(count),error_list2,"r")
            
            #if self.plot_rbf_pred and self.rbf_units==6:
            #plt.figure(512)#4*self.rbf_units+3)
            #plt.plot(self.rbf,np.zeros_like(self.rbf),"*",label="rbf units")
            #plt.plot(self.x_test,y_pred,"-",label="std={:.2f}".format(self.std)+" rbf="+str(self.rbf_units)+" "+str(self.mode))
    
            
            #elif self.plot_rbf_pred_ballist:
                #plt.figure(4*self.rbf_units+2)
                #plt.plot(range(count),error_list1,"b")
                #plt.plot(range(count),error_list2,"r")
                
          
        #if self.plot_rbf_pred and self.rbf_units==6:
            #plt.figure(512)#4*self.rbf_units+3)
            #plt.plot(self.rbf,np.zeros_like(self.rbf),"*",label="rbf units")
            #plt.plot(self.x_test,y_pred,"-",label="std={:.2f}".format(self.std)+" rbf="+str(self.rbf_units)+" "+str(self.mode))
    
            #plt.legend()
        return error_train , error_test
    
    def run_CL(self, threshold=None,transform=False, mode="batch",nr_winners=1):
        self.CL(nr_winners)   
        return self.run(threshold,transform,mode)
                        
    def CL(self,nr_winners):
        
        for j in range(10000):#(7*500):
            ind_list = [i for i in range(np.size(self.x_train))]
            random.shuffle(ind_list)
            X = self.x_train [ind_list]
            Y = self.y_train [ind_list]
            #for i in range(1):#len(X)):
            Winner=self.winner(X[0],self.rbf,nr_winners)
            for k in Winner:
                self.rbf[k]+=self.CL_eta*(X[0]-self.rbf[k])
                    
        if self.CL_std:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*self.rbf_units)
            
        
        if self.plot_rbf_pred:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.rbf,np.ones_like(self.rbf)*np.size(self.rbf),"*")

    def winner(self,x,rbf, nr_winners=1):
        distance_list=[[abs(x-rbf[i]),i] for i in range(len(rbf))]
        distance_list.sort()
        winner_list=np.array(distance_list,dtype=int)[:nr_winners][:,1]
        return winner_list


class RBFNetwork_ballist(RBFNetwork):
    
    def __init__(self, version="sin",std=None,rbf_units=5,eta=0.1, threshold=0.1, plot_rbf_pred_ballist=False, place_ver=0,
                 CL_eta=0.1, CL_plot=True):
        with open(os.path.abspath("ballist.dat")) as file: train_data=np.array([data.split() for data in file],dtype=float)
        with open(os.path.abspath("balltest.dat")) as file: test_data=np.array([data.split() for data in file], dtype=float)
        self.x_train = train_data[:,:2]
        self.x_test  = test_data[:,:2]
        self.y_train = train_data[:,2:]
        self.y_test  = test_data[:,2:]
        
        self.rbf=[np.arange(1/(2*rbf_units), 1, 1/rbf_units),np.arange(0, 1, 1/rbf_units), np.random.rand(rbf_units),
                                                                                  self.place_func(rbf_units)][place_ver]
        self.rbf=np.random.rand(2,rbf_units)#[self.rbf,self.rbf]
        self.CL_std=False
        if std==None and rbf_units==1: self.std=1.0/np.sqrt(2)
        elif std==None: 
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        else:self.std=std
        self.rbf_units=rbf_units
        self.W=np.random.rand(rbf_units,2)
        
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        self.plot_rbf_pred_ballist=plot_rbf_pred_ballist
        self.plot_rbf_pred=False
        self.CL_plot=CL_plot
        if self.plot_rbf_pred_ballist:
            plt.figure(0)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
        if self.plot_rbf_pred_ballist: 
            plt.figure(4*rbf_units)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")
            
    def calcRbf(self,x,r,s,option=0):
        if not option==1:
            return np.exp(-np.power(sum([np.transpose(np.array([x[:,i]]))-r[i] for i in range(np.shape(x)[1])]),2)/(2.0*np.power(s,2)))  
        else:
            return np.exp(-np.power(sum([np.transpose(x[i])-r[i] for i in range(int(np.size(x)))]),2)/(2.0*np.power(s,2)))  

        
    def run(self, threshold=None,transform=False, mode="batch"):
        error_train , error_test=super().run(threshold,transform, mode)
        if self.plot_rbf_pred_ballist:
            y_pred,_=self.forward(self.x_test, transform)
            plt.figure(4*self.rbf_units+3)
            plt.plot(self.y_test[:,0],self.y_test[:,1],".")
            plt.plot(y_pred[:,0],y_pred[:,1],"*")
        return error_train , error_test
    
    def winner(self,x,rbf):
        return np.argmin(np.sqrt(np.power(x[0]-rbf[0],2)+np.power(x[1]-rbf[1],2)))

    def CL(self,nr_winners):
        #add more winners
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        for j in range(1000):#(7*500):
            #for i in range(len(X)):
            Winner=self.winner(X[0],self.rbf)
            self.rbf[:,Winner]+=self.CL_eta*(X[0]-self.rbf[:,Winner])
        if self.CL_std:
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*self.rbf_units)
            
        if self.CL_plot:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")


# In[88]:


#3.2 plot rbf and width

for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=0.1#/np.sqrt(2)#1 #None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    threshold=0
    
    for std_ in [0.1, 10]:#, 0.01, 0.001, 0]:
        #if  transform_==False and threshold==0 or transform_==True and threshold!=0:
        #    continue
        for count in [2,10]:#error_test> threshold:
            rbf_NN=RBFNetwork(rbf_units=count, std=std_, version=func, place_ver=0, plot_rbf_pred=True,use_seed=True)
            rbf_NN.add_noise()
            error_train , error_test=rbf_NN.run(transform=transform_)
            error_list1+=[error_train]
            error_list2+=[error_test]
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
                  "\nnr of units",count,"\n-------------")
        #plt.figure(2000)
        #plt.plot(range(4),error_list1,"b")
        #plt.plot(range(4),error_list2,"r")
    print("\n")
    plt.figure(512).suptitle('batch vs online predictions for square(2x)')
    plt.xlabel('input')
    plt.ylabel('ouput')
    plt.figure(512).savefig('square2x.jpg')


# In[21]:


#3.2  noise batch
import numpy as np

for i in range(1):
    func=["sin", "square", "square"][0]
    transform_=[False, False, True][0]
    print(func+"2x | transform = "+str(transform_))
    std_=None#/np.sqrt(2)#1 #None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    
    for threshold in [0.1, 0]:#, 0.01, 0.001, 0]:
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while True:#error_test> threshold:
            count+=1
            #t1=time.time()
            rbf_NN=RBFNetwork(rbf_units=count, std=std_, version=func, place_ver=2, plot_rbf_pred=True,use_seed=True,rd_seed=119)
            #rbf_NN.add_noise(test=False)
            xtr,xte,ytr,yte=rbf_NN.add_noise()
            if count==43:
                x_train2=xtr
                x_test2=xte
                y_train2=ytr
                y_test2=yte
                
            error_train , error_test=rbf_NN.run(transform=transform_)
            #t2=time.time()
            #print("timebatch",t2-t1)
            error_list1+=[error_train]
            error_list2+=[error_test]
            if count>42:#63:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
        plt.figure(2000*i)
        plt.plot(range(1,count+1),error_list1,"b")
        plt.plot(range(1,count+1),error_list2,"r")
    print("\n")

    
    
#import time
#x_train2=x_train2.reshape(-1,1)
#y_train2=y_train2.reshape(-1,1)
#x_test2=x_test2.reshape(-1,1)
#y_test2=y_test2.reshape(-1,1)
#print(np.shape(x_train2))
#print(np.shape(y_train2))
#print(type(x_train2))
#print(type(y_train2))
#error_lt=[]
#for i in range(10):
#    mlpNN=MLP(learning_rate=0.1,epochs=100,hidden_nodes=6)
#    mlpNN.learn(x_train2,y_train2)
#    pred=mlpNN.predict(patterns=x_test2)
#    error_lt+=[np.mean(abs(pred - y_test2))]
    
    
    
    
#print(pred)

#print(np.mean(abs(pred - y_test2)))
#plt.figure(512)
#plt.plot(x_test2,pred,"-",label="mlp_with_6_units" )
#plt.legend()
#print(np.mean(error_lt),np.std(error_lt))
#plt.figure(512).suptitle('rbf vs mlp predeictions for sin(2x)')
#plt.xlabel('input')
#plt.ylabel('ouput')
#plt.figure(512).savefig('mlp vs batch for sin 2x.jpg')

#________________________________________
#test error 0.11540912859900067  6 units

#test error 0.2120452068964782 
#test error 0.11922598873878443 
#test error 0.1840228475494015 

#__________________________________
#0.1303702691553523  units 20

#0.1361350159467203
#0.12959886596918982
#0.13429880638489264


#___________________________________
#0.312121986353901  43

#0.38282283933145517
#0.17753097151252498 
#0.18505365127582776 


# In[116]:


import time
x_train2=x_train2.reshape(-1,1)
y_train2=y_train2.reshape(-1,1)
x_test2=x_test2.reshape(-1,1)
y_test2=y_test2.reshape(-1,1)
#print(np.shape(x_train2))
#print(np.shape(y_train2))
#print(type(x_train2))
#print(type(y_train2))
error_lt=[]
for i in range(10):
    mlpNN=MLP(learning_rate=0.1,epochs=100,hidden_nodes=27)
    mlpNN.learn(x_train2,y_train2)
    pred=mlpNN.predict(patterns=x_test2)
    error_lt+=[np.mean(abs(pred - y_test2))]


# In[117]:


#print(pred)

#print(np.mean(abs(pred - y_test2)))
plt.figure(512)
plt.plot(x_test2,pred)
print(np.mean(error_lt),np.std(error_lt))


# In[7]:


#std=2*pi

print("min error",min(error_list2))
print("idx",np.argmin(error_list2))
print("error",error_list2[7])


#sin(2x)
#noise 
#min error 0.12173189111388509   
#idx 7

#random
#min error 0.1209066971109389           0.204845442197182(7)
#idx 10

#noise only train
#min error 0.042689733943402204        0.043688709628892784 (7)
#idx 10

#clean
#min error 0.028357308155354612
#idx  7


#square(2x)
#noise
#min error 0.3226210989828487
#idx 7

#random
#min error 0.322367270237809          0.4361799555370918
#idx 10

#noise only train
#min error 0.29164244636656744        0.29164244636656744
#idx 7

#clean
#min error 0.2855989970858135
#idx 7


# In[36]:


#std=None

print("min error",min(error_list2))
print("idx",np.argmin(error_list2))
print("error",error_list2[5])


#sin(2x)
#noise
#min error 0.11540912859900067    perception  mean 0.13871214778363122   std 0.0007348419573644233
#idx 5

#random
#min error 0.12110360079114833    0.2927422290254023
#idx 10

#noise only train
#min error 0.024289403729426257 
#idx 5

#clean
#min error 8.955073310354125e-07
#idx 19


#square(2x)
#noise
#min error 0.22880022861124563    mean 0.1551909057758762   std 0.0005552761261282593
#idx 22


#random
#min error 0.21971022745661967     0.23572257218173945
#idx 36

#noise only train
#min error 0.15677570161365328     0.17348690260024296(22)
#idx 28

#clean
#min error 0.13823421417720735
#idx 37


# In[19]:


#std=0.1

print("min error",min(error_list2))
print("idx",np.argmin(error_list2))
print("error",error_list2[26])

#sin(2x)
#noise
#min error 0.1324008382029413<-
#idx 26

#random
#min error 0.3279066925899534
#idx 34

#noise only train
#min error 0.08132121254390945  0.09440644484078076(26)<-
#idx 28

#clean
#min error 0.0012389169676075289
#idx idx 59


#square(2x)
#noise
#min error 0.22641605318013935<-       mean 0.15493917563320092 std 0.0004943348151159742
#idx 26

#random
#min error 0.6030707383468643
#idx 32

#noise only train
#min error 0.15762788321829668   0.16476324822773938<-
#idx 27

#clean
#min error 0.08416940961147136
#idx 31


# In[23]:


#std=10
#noise
#noise only train
#clean
print("min error",min(error_list2))
print("idx",np.argmin(error_list2))
print("error",error_list2[53])

#sin(2x)
#noise
#min error 0.11988169401074052
#idx 53

#random
#min error 0.1201384291917921
#idx 41

#noise only train
#min error 0.04300617974322364  0.04300617974322364
#idx 53  

#clean
#min error 0.031481122570578945
#idx 48


#square(2x)
#noise
#min error 0.3224820744602792
#idx 14

#random
#min error 0.3226990883491681  
#idx 55

#noise only train
#min error 0.2914186507936508  0.29377480158730157(14)
#idx 11

#clean
#min error 0.2855902777777778
#idx 16


# In[221]:


#std=1
print("min error",min(error_list2))
print("idx",np.argmin(error_list2))

#sin(2x)

#noise
#min error 0.12155863297226262
#idx 10 ->>>>>

#random
#min error 0.12148378080349327
#idx 8

#noise only train
#min error 0.04539404504556223
#idx 10

#clean
#min error 2.645013982842084e-06
#idx 17



#square(2x)

#noise
#min error 0.23186027988805052
#idx 17

#noise only train
#min error 0.17623544144370254
#idx 17

#clean
#min error 0.1765794905405196
#idx 25


#square(2x) transform

#noise
# imposible to get 0 because the outpurs are 1 and -1 while the noise give the data other values
#min error 0.15370124290481155
#idx 5

#random
#min error 0.23189805715343617
#idx 19

#noise only train
#min error 0.0
#idx 5

#clean
#min error 0.0
#idx 5



#___________________________________

#min error 0.12389881705035555
#idx 15

#min error 0.13092104160329704
#idx 12

#min error 0.1255023860647029
#idx 6

#min error 0.12375484238893773
#idx 13
#_________________________________

#min error 0.12123776484130756
#idx 8

#min error 0.14600615352367705
#idx 7


# In[ ]:


#3.2 noise online 0.1 askjdlasjdljasdklösa
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    for threshold in [0.1]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            rbf_NN.add_noise()
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            if count>20:
                break
            error_list1+=[error_train]
            error_list2+=[error_test]
        
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
        plt.figure(1000*i)
        plt.plot(range(1,count+1),error_list1,"b")
        plt.plot(range(1,count+1),error_list2,"r")
    print("\n")


# In[30]:


#3.2 noise online 0.1 this one
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=1#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    for threshold in [0.01]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=10
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0, plot_rbf_pred=True)
            rbf_NN.add_noise()
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            if count>63:
                break
            error_list1+=[error_train]
            error_list2+=[error_test]
        
            print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
                  "\nnr of units",count,"\n-------------")
        #plt.figure(1000*i)
        #plt.plot(range(1,count),error_list1,"b")
        #plt.plot(range(1,count),error_list2,"r")
    print("\n")


# In[31]:


#3.2 noise online 0.1 this one
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=1#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    for threshold in [0.01]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=10
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.01, std=std_, version=func, place_ver=0, plot_rbf_pred=True)
            rbf_NN.add_noise()
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            if count>63:
                break
            error_list1+=[error_train]
            error_list2+=[error_test]
        
            print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
                  "\nnr of units",count,"\n-------------")
        #plt.figure(1000*i)
        #plt.plot(range(1,count),error_list1,"b")
        #plt.plot(range(1,count),error_list2,"r")
    print("\n")


# In[44]:


#3.2 noise online 0.1 this one
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    error_list1=[]
    error_list2=[]
    threshold=0.0001
    for count in [6 ,10]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        #while error_test> threshold:
        #count+=0
        rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.01, std=std_, version=func, place_ver=0, plot_rbf_pred=True)
        rbf_NN.add_noise()
        error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
        #if count>63:
        #    break
        error_list1+=[error_train]
        error_list2+=[error_test]

        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
        #plt.figure(1000*i)
        #plt.plot(range(1,count),error_list1,"b")
        #plt.plot(range(1,count),error_list2,"r")
    print("\n")


# In[43]:


sin2x | transform = False

threshold 0.0001 
train error 0.12494817599033632 
test error 0.11939003816354897 
nr of units 6 
-------------

threshold 0.0001 
train error 0.13157899408782017 
test error 0.1252315437518093 
nr of units 10 

plt.plot(range(1,count),error_list1,"b")
plt.plot(range(1,count),error_list2,"r")


# In[8]:


print(min(error_list1))
print(np.argmin(error_list1))


# In[19]:


print(np.exp(-1))
print(np.exp(-8))
print(np.exp(-0.01))
print(np.exp(-0.08))


# In[13]:


#3.2 noise online 0.1 for longer 
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    for threshold in [0.1]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            count=8
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            rbf_NN.add_noise(test=False)
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            if count>1:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[16]:


#check 0.001 clean online

#3.2
for i in range(1):
    func=["sin", "square", "square"][i]
    transform_=[False, False, True][i]
    print(func+"2x | transform = "+str(transform_))
    std_=None#/np.sqrt(2)# None
    count=0
    error_train=np.inf
    error_test=np.inf
    for threshold in [0.001]:#,0.01, 0.001, 0]:
        #threshold=0.1
        if  transform_==False and threshold==0 or transform_==True and threshold!=0:
            continue
        while error_test> threshold:
            count+=1
            count=10
            rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, version=func, place_ver=0)#, plot_rbf_pred=True)
            #rbf_NN.add_noise(test=False)
            error_train , error_test=rbf_NN.run(mode="online",transform=transform_)
            if count>1:
                break
        print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
              "\nnr of units",count,"\n-------------")
    print("\n")


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random 
import os

class RBFNetwork():
    
    def __init__(self, version="sin",std=None,rbf_units=4,eta=0.1, threshold=0.1, plot_rbf_pred=False, place_ver=0,CL_eta=0.1):
        self.x_train = np.array([np.arange(0, 2*np.pi, 0.1)])
        self.x_test  = np.array([np.arange(0.05, 2*np.pi, 0.1)])
        self.x_train = np.arange(0, 2*np.pi, 0.1)
        self.x_test  = np.arange(0.05, 2*np.pi, 0.1)
        self.y_train = np.sin(2*self.x_train)
        self.y_test  = np.sin(2*self.x_test)        
        if version=="square":            
            self.y_train = np.where(np.sign(self.y_train)>=0, 1.0, -1.0,)
            self.y_test  = np.where(np.sign(self.y_test)>=0, 1.0, -1.0)
            
        self.rbf=[np.arange(2*np.pi/(2*rbf_units), 2*np.pi, 2*np.pi/rbf_units),np.arange(0, 2*np.pi, 2*np.pi/rbf_units),
                                                          np.random.rand(rbf_units)*(2*np.pi),self.place_func(rbf_units)][place_ver]    
        
        self.CL_std=False
        if std==None and rbf_units!=1:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        elif std==None and rbf_units==1:
            self.std=1/np.sqrt(2*rbf_units)
        
        else:
            self.std=std
            
        self.W=np.random.rand(rbf_units)
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        self.plot_rbf_pred=plot_rbf_pred
        self.plot_rbf_pred_ballist=False
        if self.plot_rbf_pred:
            plt.figure(0)
            plt.plot(self.rbf,np.ones_like(self.rbf)*rbf_units,"*")
            
    def place_func(self,nr_units):
        place_list=np.arange(np.pi/4.0, 2*np.pi, np.pi/2.0)[0:nr_units]
        if nr_units>4:
            list_rbf=[2*np.pi]
            list_rbf2=np.arange(0, 2*np.pi, 2*np.pi/(max(nr_units-5,1.0)))
            for i in place_list: list_rbf+=[i]
            
            if nr_units>5:
                for i in list_rbf2:  list_rbf+=[i]
            place_list=np.array(list_rbf)
        return place_list

    def plot_data(self):
        plt.plot(self.x_train,self.y_train,"*")
        plt.plot(self.x_test,self.y_test,"*")

    def add_noise(self, test=True):
        #np.size(self.x_train)
        self.x_train +=np.random.normal(0,0.1,np.shape(self.x_train))
        self.x_test +=np.random.normal(0,0.1,np.shape(self.x_train))
        if test:
            self.y_train +=np.random.normal(0,0.1,np.shape(self.x_train))
            self.y_test +=np.random.normal(0,0.1,np.shape(self.x_train))

    def calcRbf(self,x,r,s,option=0):
        #if self.mode=="batch":
        return np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2)))              
        #else:
        #    return np.exp(-np.power(x-r,2)/(2.0*np.power(s,2)))
    
    def forward(self, X, transform, option=0): 
        phi=self.calcRbf(X,self.rbf,self.std, option)
        #y_pred=np.dot(self.W,np.transpose(phi))
        y_pred=np.dot(phi,self.W)
        if transform:
                y_pred = np.sign(y_pred)
        return y_pred , phi
          
    def batch_mode_training(self,transform):
        phi=self.calcRbf(self.x_train,self.rbf,self.std)
        self.W = np.dot(np.linalg.pinv(phi),self.y_train)
        #y_pred=np.dot(self.W,np.transpose(phi))
        y_pred=np.dot(phi,self.W)
        if transform:
            y_pred = np.sign(y_pred)        
        error=np.mean(abs(self.y_train-y_pred))        
        return error
            
    def on_line_learning(self,transform):
        # one epoch
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        error_list=[]
        for i in range(len(X)):
            x=X[i]
            y=Y[i]
            y_pred, phi=self.forward(x, transform, option=1)
            e=y-y_pred
            #error_list+=[abs(e)]
            error_list+=[np.sqrt(np.dot(e,e))]
            if np.size(e)==1:
                delta_W=self.eta*e*phi
            else:
                delta_W=np.array([self.eta*e[i]*phi for i in range(np.size(e))]).T
            self.W += delta_W
        error=np.sqrt(np.dot(e,e))#abs(e)#np.mean(error_list)
        return error
                    
    def run(self, threshold=None,transform=False, mode="batch"):
        self.mode=mode
        if threshold==None:
            threshold=self.threshold
            
        if self.mode=="batch":
            error_train=self.batch_mode_training(transform)
            y_pred,_=self.forward(self.x_test, transform)  
            error_test=np.mean(abs(self.y_test-y_pred))  
            
        else:
            error=np.inf
            error_train=99999999999
            error_test=99999999999
            count=0
            error_list1=[]
            error_list2=[]
            conditional_statement=True
            while count<(np.size(self.rbf)*1000) and error_test>self.threshold and conditional_statement:# (error-error_test)>1e-180 and count < 999999 and error>error_test:
                count+=1
                #if count==1000:
                #    self.eta=self.eta/10.0
                #error=error_test
                #old_error=error_test
                #old_weigths=self.W
                error_train=self.on_line_learning(transform)
                y_pred,_=self.forward(self.x_test, transform)
                error_test=np.mean(abs(self.y_test-y_pred))
                error_list1+=[error_train]
                error_list2+=[error_test]
                #print("count",count,"error",error_test)
                #if count>1000:
                #    if np.mean(error_list2[-1000:-500])<(np.mean(error_list2[-500:])+1e-11):
                #        conditional_statement=False                    
            
            if self.plot_rbf_pred:
                plt.figure(np.size(self.rbf))
                plt.plot(range(count),error_list1,"b")
                plt.plot(range(count),error_list2,"r")
            
            elif self.plot_rbf_pred_ballist:
                plt.figure(4*self.rbf_units+2)
                plt.plot(range(count),error_list1,"b")
                plt.plot(range(count),error_list2,"r")
            
            #self.W=old_weigths
            #error_test=error
            #plt.figure(count)
            
            
                
        if self.plot_rbf_pred:
            plt.figure(np.size(self.rbf)+2)
            plt.plot(self.rbf,np.zeros_like(self.rbf),"+")
            plt.plot(self.x_test,y_pred,"-")
        return error_train , error_test
    
    def run_CL(self, threshold=None,transform=False, mode="batch",nr_winners=1):
        self.CL(nr_winners)   
        return self.run(threshold,transform,mode)
                        
    def CL(self,nr_winners):
        ind_list = [i for i in range(np.size(self.x_train))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        for j in range(1000):#(7*500):
            for i in range(len(X)):
                Winner=self.winner(X[i],self.rbf,nr_winners)
                for k in Winner:
                    self.rbf[k]+=self.CL_eta*(X[i]-self.rbf[k])
                    
        if self.CL_std:
            d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            
        plt.figure(1)
        if self.plot_rbf_pred:
            plt.plot(self.rbf,np.ones_like(self.rbf)*np.size(self.rbf),"*")

    def winner(self,x,rbf, nr_winners=1):
        #if nr_winners==1:
        #    return np.argmin(abs(x-[rbf]))
        #else:
        distance_list=[[abs(x-rbf[i]),i] for i in range(len(rbf))]
        distance_list.sort()
        winner_list=np.array(distance_list,dtype=int)[:nr_winners][:,1]
        return winner_list


class RBFNetwork_ballist(RBFNetwork):
    
    def __init__(self, version="sin",std=None,rbf_units=5,eta=0.1, threshold=0.1, plot_rbf_pred_ballist=False, place_ver=0,
                 CL_eta=0.1, CL_plot=True):
        with open(os.path.abspath("ballist.dat")) as file: train_data=np.array([data.split() for data in file],dtype=float)
        with open(os.path.abspath("balltest.dat")) as file: test_data=np.array([data.split() for data in file], dtype=float)
        self.x_train = train_data[:,:2]
        self.x_test  = test_data[:,:2]
        self.y_train = train_data[:,2:]
        self.y_test  = test_data[:,2:]
        
        self.rbf=[np.arange(1/(2*rbf_units), 1, 1/rbf_units),np.arange(0, 1, 1/rbf_units), np.random.rand(rbf_units),
                                                                                  self.place_func(rbf_units)][place_ver]
        self.rbf=np.random.rand(2,rbf_units)#[self.rbf,self.rbf]
        self.CL_std=False
        if std==None and rbf_units==1: self.std=1.0/np.sqrt(2)
        elif std==None: 
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*rbf_units)
            self.CL_std=True
        else:self.std=std
        self.rbf_units=rbf_units
        self.W=np.random.rand(rbf_units,2)
        
        self.eta=eta
        self.CL_eta=CL_eta
        self.threshold=threshold
        self.mode="batch"
        self.plot_rbf_pred_ballist=plot_rbf_pred_ballist
        self.plot_rbf_pred=False
        self.CL_plot=CL_plot
        if self.plot_rbf_pred_ballist:
            plt.figure(0)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
        if self.plot_rbf_pred_ballist: 
            plt.figure(4*rbf_units)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")
            
    def calcRbf(self,x,r,s,option=0):
        if not option==1:
            return np.exp(-np.power(sum([np.transpose(np.array([x[:,i]]))-r[i] for i in range(np.shape(x)[1])]),2)/(2.0*np.power(s,2)))  
        else:
            return np.exp(-np.power(sum([np.transpose(x[i])-r[i] for i in range(int(np.size(x)))]),2)/(2.0*np.power(s,2)))  

        
    def run(self, threshold=None,transform=False, mode="batch"):
        error_train , error_test=super().run(threshold,transform, mode)
        if self.plot_rbf_pred_ballist:
            y_pred,_=self.forward(self.x_test, transform)
            plt.figure(4*self.rbf_units+3)
            plt.plot(self.y_test[:,0],self.y_test[:,1],".")
            plt.plot(y_pred[:,0],y_pred[:,1],"*")
        return error_train , error_test
    
    def winner(self,x,rbf):
        return np.argmin(np.sqrt(np.power(x[0]-rbf[0],2)+np.power(x[1]-rbf[1],2)))

    def CL(self):
        ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
        random.shuffle(ind_list)
        X = self.x_train [ind_list]
        Y = self.y_train [ind_list]
        for j in range(1000):#(7*500):
            for i in range(len(X)):
                Winner=self.winner(X[i],self.rbf)
                self.rbf[:,Winner]+=self.CL_eta*(X[i]-self.rbf[:,Winner])
        if self.CL_std:
            d=max([np.sqrt(np.dot(self.rbf[:,i]-self.rbf[:,j],self.rbf[:,i]-self.rbf[:,j])) for i in range(len(self.rbf[0]))  
                                                                                    for j in range(len(self.rbf[0])) if i>j])
            self.std=d/np.sqrt(2*self.rbf_units)
            
        if self.CL_plot:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.x_train[:,0],self.x_train[:,1],"+")
            plt.plot(self.rbf[0],self.rbf[1],"*")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

class RBFNetwork():
    
    def __init__(self, version="sin",std=1,rbf_units=60,eta=0.01, threshold=0.1):
        self.x_train = np.array([np.arange(0, 2*np.pi, 0.1)])
        self.x_test  = np.array([np.arange(0.05, 2*np.pi, 0.1)])
        self.x_train = np.arange(0, 2*np.pi, 0.1)
        self.x_test  = np.arange(0.05, 2*np.pi, 0.1)
        self.y_train = np.sin(2*self.x_train)
        self.y_test  = np.sin(2*self.x_test)
        if version=="square":            
            self.y_train = np.where(np.sign(self.y_train)>=0, 1, -1)
            self.y_test  = np.where(np.sign(self.y_test)>=0, 1, -1)
        self.std=std       
        self.rbf=np.arange(0, 2*np.pi, 2*np.pi/rbf_units)
        self.W=np.random.rand(rbf_units)
        self.eta=eta
        self.threshold=threshold
        self.mode="batch"

    def plot_data(self):
        plt.plot(self.x_train,self.y_train,"*")
        plt.plot(self.x_test,self.y_test,"*")

    def add_noise(self, test=True):
        self.x_train +=np.random.normal(0,0.1,np.size(self.x_train))
        self.x_test +=np.random.normal(0,0.1,np.size(self.x_test))
        if test:
            self.y_train +=np.random.normal(0,0.1,np.size(self.y_train))
            self.y_test +=np.random.normal(0,0.1,np.size(self.y_test))

    def calcRbf(self,x,r,s):
        
        if self.mode=="batch":
            return np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(s,2)))              
        else:
            return np.exp(-np.power(x-r,2)/(2.0*np.power(s,2)))
    
    def forward(self, X): 
        phi=self.calcRbf(X,self.rbf,self.std)
        y_pred=np.dot(self.W,np.transpose(phi))
        return y_pred , phi
          
    def batch_mode_training(self,transform):
        y_pred, phi=self.forward(self.x_train)
        if transform:
            y_pred = np.sign(y_pred)        
        e=(self.y_train-y_pred)
        error=np.mean(abs(e))           
        delta_W=np.dot(self.eta*e,phi)/np.size(e)
        self.W += delta_W
        return error
            
    def on_line_learning(self,transform):
        # one epoch 
        for x in self.x_train:
            y_pred, phi=self.forward(x)
            if transform:
                y_pred = np.sign(y_pred)
            print("1",np.shape(self.y_train))
            print("2",np.shape(y_pred))
            print("3",np.shape(self.y_train[0]))
            e=(self.y_train-y_pred)
            print("haha",np.shape(e))
            error=np.mean(abs(e))           
            delta_W=np.dot(self.eta*e,phi)/np.size(e)
            self.W += delta_W
        return error
                    
    def run(self, threshold=None,transform=False,mode="batch"):
        self.mode=mode
        if threshold==None:
            threshold=self.threshold
        error=np.inf
        count=0
        while error>threshold and count < 99999:
            count+=1
            if self.mode=="batch":
                error=self.batch_mode_training(transform)
            else:
                error=self.on_line_learning(transform)
            
        print("iter",count)
        print("error", error)

    def winner(self,x,rbf):
        return np.argmin(abs(np.transpose(x)-rbf),axis=1)

