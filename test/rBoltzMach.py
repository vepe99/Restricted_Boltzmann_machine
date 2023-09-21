import numpy as np

class boltzmanmachine:
    
    def __init__(self, L, M, sigma, De=2, vmin=-1, spin=True):
        '''
        Initialize the object with defaults parameter which are determined by
        meta reciew.
        w: network weights
        a: visible bias
        b: hidden bias
        spin: set the notation used by the model, (0,1) or (-1,1)
        DE: energy separation among the two possible states allowed in the positive phases, 1 or 2
        vmin: minimum value of the representation
        '''
        self.w = np.random.normal(loc=0.0, scale=sigma, size=(L,M))
        self.a = np.random.normal(loc=0.0, scale=sigma, size=L)
        self.b = b = np.zeros(M)
        
        self.spin=spin
        
        self.DE = De
        self.L = int(L)
        self.M = int(M)
        self.vmin=vmin
        
        self.v_data, self.v_model = None, None
        self.h_data, self.h_model = None, None
        self.vh_data,self.vh_model= None, None
        
        #one hot encoding of the four possible states
        if self.spin:
            self.csi1 = np.array([1,-1,-1,-1])    
            self.csi2 = np.array([-1,1,-1,-1])   
            self.csi3 = np.array([-1,-1,1,-1])    
            self.csi4 = np.array([-1,-1,-1,1])  
        else:
            self.csi1 = np.array([1,0,0,0])    
            self.csi2 = np.array([0,1,0,0])   
            self.csi3 = np.array([0,0,1,0])    
            self.csi4 = np.array([0,0,0,1]) 
            
        self.csi = [self.csi1, self.csi2, self.csi3, self.csi4]
        
        #algotithm variables
        self.sa_t0 = 0
        self.sb_t0 = 0
        self.sw_t0 = 0
        
        self.ma_t0 = 0
        self.mb_t0 = 0
        self.mw_t0 = 0
        
        self.batch_counter = 0
        
    def load_model(self, file_name):
        model = np.load(file_name)
        
        self.w = model['w']
        self.a = model['a']
        self.b = model['b']
        
        
        
    def save_model(self, file_name):
        np.savez(file_name, a=self.a, b=self.b, w=self.w)
        
        
        
    def init_avg(self):
        '''
        Set  to zero the averages quantities needed to compute the gradien
        '''
        self.v_data, self.v_model = np.zeros(self.L),np.zeros(self.L)
        self.h_data, self.h_model = np.zeros(self.M),np.zeros(self.M)
        self.vh_data,self.vh_model= np.zeros((self.L,self.M)),np.zeros((self.L,self.M))
        
        
        
    def positive(self, v_in, Amp=1.):
        '''
        Positive phase of the training
        Visible -> Hidden
        No one-hot encoding needed
        '''
        act = np.dot(v_in, self.w) + self.b      
        #print(act)
        argument=np.exp(-Amp*self.DE*act)
        prob = 1. / (1. + argument)
        n = np.shape(act)
        h = np.full(n, self.vmin, dtype=int) # a list on -1's or 0's
        h[np.random.random_sample(n) < prob] = 1
        
        return h
    
    def neg(self, h_in, Amp=1.):
        '''
        Negative phase of the training
        Hidden -> Visible
        No one-hot encoding needed
        '''
        act = np.dot(h_in, self.w.T) + self.a      
        #print(act)
        prob = 1. / (1. + np.exp(-Amp*self.DE*act))
        n = np.shape(act)
        vf = np.full(n, self.vmin, dtype=int) # a list on -1's or 0's
        vf[np.random.random_sample(n) < prob] = 1
        
        return vf
    
    
    def negative(self, h_in, Amp=1.):
        '''
        Negative phase of the training
        Hidden -> Visible
        With ne-hot encoding needed
        '''
        
        weigths = np.reshape(np.dot(h_in, self.w.T) + self.a, (5,4) ) 

        
        E1 = np.dot(weigths, self.csi1) #array of length 5, the number of amminoacids
        E2 = np.dot(weigths, self.csi2)
        E3 = np.dot(weigths, self.csi3)
        E4 = np.dot(weigths, self.csi4)
        


        Z = np.exp(-Amp*E1) + np.exp(-Amp*E2) + np.exp(-Amp*E3) + np.exp(-Amp*E4) #partition function for each amminoacid
        
        p1 = np.exp(-Amp*E1)/Z 
        p2 = np.exp(-Amp*E2)/Z
        p3 = np.exp(-Amp*E3)/Z
        p4 = np.exp(-Amp*E4)/Z
        
        
        
        p = np.reshape(np.concatenate((p1, p2, p3, p4)), (4, 5))      

        
        cum = np.cumsum(p, axis=0) #(4x5) containing the comulatives  
        r = np.random.random(size=5)        
        
        mask = cum < r    
        indx = []    
        
        for i in range(mask.shape[1]):
            __, index = np.unique(mask[:, i], return_index=True)
            indx.append(index[0])
            
        vf=np.concatenate((self.csi[indx[0]], self.csi[indx[1]], self.csi[indx[2]], self.csi[indx[3]], self.csi[indx[4]]))
        return vf
        
    
    def update_vh(self, v_k, vf, h, hf, mini):
        '''
        Update the averages needed to compute the gradient
        '''
        self.v_data  += v_k/mini
        self.v_model += vf/mini
        self.h_data  += h/mini
        self.h_model += hf/mini
        self.vh_data += np.outer(v_k.T,h)/mini
        self.vh_model+= np.outer(vf.T,hf)/mini
    
    def SGD(self, l_rate_m):
        '''
        Stochastic gradient descent algorithm
        '''
        dw = l_rate_m*(self.vh_data - self.vh_model)
        da = l_rate_m*(self.v_data - self.v_model)
        db = l_rate_m*(self.h_data - self.h_model)
        
        self.w += dw
        self.a += da
        self.b += db
        
    def RMSprop(self, eta_t, beta=0.9, epsilon=1e-8):
        '''
        RMSprop algorithm
        '''
        ga_t = self.v_data - self.v_model
        gb_t = self.h_data - self.h_model
        gw_t = self.vh_data - self.vh_model
        
        sa_t = beta*self.sa_t0 + (1-beta)*ga_t**2
        sb_t = beta*self.sb_t0 + (1-beta)*gb_t**2
        sw_t = beta*self.sw_t0 + (1-beta)*gw_t**2
        
        self.sa_t0 = sa_t
        self.sb_t0 = sb_t
        self.sw_t0 = sw_t
        
        
        self.a = self.a + eta_t*ga_t/np.sqrt(sa_t + epsilon)
        self.b = self.b + eta_t*gb_t/np.sqrt(sb_t + epsilon)
        self.w = self.w + eta_t*gw_t/np.sqrt(sw_t + epsilon)
        
        
    def ADAM(self, eta_t, epoch, beta1=0.9, beta2=0.99,epsilon=1e-8):
        '''
        ADAM algorithm
        '''
        ga_t = self.v_data - self.v_model
        gb_t = self.h_data - self.h_model
        gw_t = self.vh_data - self.vh_model

        ma_t = beta1*self.ma_t0 + (1-beta1)*ga_t
        mb_t = beta1*self.mb_t0 + (1-beta1)*gb_t
        mw_t = beta1*self.mw_t0 + (1-beta1)*gw_t

        sa_t = beta2*self.sa_t0 + (1-beta2)*ga_t**2
        sb_t = beta2*self.sb_t0 + (1-beta2)*gb_t**2
        sw_t = beta2*self.sw_t0 + (1-beta2)*gw_t**2

        self.sa_t0 = sa_t
        self.sb_t0 = sb_t
        self.sw_t0 = sw_t

        self.ma_t0 = ma_t
        self.mb_t0 = mb_t
        self.mw_t0 = mw_t

        ma_t_hat = ma_t/(1-beta1**epoch) 
        mb_t_hat = mb_t/(1-beta1**epoch) 
        mw_t_hat = mw_t/(1-beta1**epoch)

        sa_t_hat = sa_t/(1-beta2**epoch) 
        sb_t_hat = sb_t/(1-beta2**epoch) 
        sw_t_hat = sw_t/(1-beta2**epoch)

        self.a = self.a + eta_t*ma_t_hat/(np.sqrt(sa_t_hat) + epsilon)
        self.b = self.b + eta_t*mb_t_hat/(np.sqrt(sb_t_hat) + epsilon)
        self.w = self.w + eta_t*mw_t_hat/(np.sqrt(sw_t_hat) + epsilon)
        
    def train(self, data, learning_rate, batch_size, n_contrastive_div, Amp_training, Algorithm, epoch=1):
        
        if self.batch_counter == 0:
            self.init_avg()
            
        v_k = np.copy(data)
        vf = np.copy(data)
        
        for i in np.arange(n_contrastive_div):
            h = self.positive(vf, Amp_training)
            vf = self.negative(h, Amp_training)
        hf = self.positive(vf, Amp_training)
        
        self.update_vh(v_k, vf, h, hf, batch_size)
        
        self.batch_counter += 1
        
        if self.batch_counter == batch_size:
            if Algorithm == 'SGD':
                self.SGD(learning_rate)
            if Algorithm == 'RMSprop':
                self.RMSprop(learning_rate)
            if Algorithm == 'Adam':
                self.ADAM(l_rate, epoch+1)
                
            self.batch_counter = 0


    def gen_fantasy(self, data, Amp_gen):
        
        vf = np.zeros_like(data)
        N = data.shape[0]
        
        for k in range(N):
            # positive CD phase: generating h 
            h = self.positive(data[k],Amp_gen)
            # negative CD phase: generating fantasy vf with low T == large GAP
            vf[k] = self.negative(h,Amp_gen)
            
        return vf
