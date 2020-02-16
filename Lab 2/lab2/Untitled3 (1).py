#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
            return np.exp(-np.power(np.transpose(np.array([x]))-r,2)/(2.0*np.power(self.std_list,2)))
    
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
                    
        #if self.CL_std:
        #    d=max([abs(self.rbf[i]-self.rbf[j]) for i in range(len(self.rbf))  for j in range(len(self.rbf)) if i>j])
        #    self.std=d/np.sqrt(2*self.rbf_units)
            
        
        if self.plot_rbf_pred:
            plt.figure(4*self.rbf_units+1)
            plt.plot(self.rbf,np.ones_like(self.rbf)*np.size(self.rbf),"*")
        
        
        distance_rx_list=[]
        for i in range(len(self.x_train)):
            distance_rx_list+=list(self.winner(self.x_train[i],self.rbf,nr_winners=1))
        distance_rx_list=np.array(distance_rx_list)
        std_list=[]
        for i in range(self.rbf_units):
            std_list+=[np.sqrt(np.mean(np.power(self.x_train[np.where(np.array(distance_rx_list)==i)]-self.rbf[i],2)))]
        self.std_list=np.array(std_list)
        self.cl_std=True
                
            
        
            
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
        for j in range(100000):#(7*500):
            ind_list = [i for i in range(int(np.size(self.x_train)/np.size(self.x_train[0])))]
            random.shuffle(ind_list)
            X = self.x_train [ind_list]
            Y = self.y_train [ind_list]
            #for j in range(10000):#(7*500):
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


# In[ ]:


#3.3.3
std_=None
count=0
error_train=np.inf
error_test=np.inf
error_list1=[]
error_list2=[]
threshold=0.001#,0.01, 0.001, 0]:
count=6
rbf_NN=RBFNetwork(rbf_units=count,threshold=threshold ,eta=0.1, std=std_, place_ver=2,plot_rbf_pred=True)
rbf_NN.add_noise()
error_train , error_test=rbf_NN.run_CL(mode="online",nr_winners=1)
error_list1+=[error_train]
error_list2+=[error_test]

#plt.figure(0)
#plt.plot(range(1,count+1),error_list1,"b")
#plt.plot(range(1,count+1),error_list2,"r")
print("\nthreshold", threshold,"\ntrain error",error_train,"\ntest error",error_test,
      "\nnr of units",count,"\n-------------")
print("\n")

