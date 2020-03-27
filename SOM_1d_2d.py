import numpy as np


class SOM_1D(object):
    
    def __init__(self, n_features, n_input):
        
        self.n_features = n_features  # number of output neurons
        self.n_input = n_input  # number of input dimension
        
        # initialize the synaptic weights, with N(0,1)
        self._W = np.random.normal(0,1,size=(n_features, n_input))
        
        # initialize the list to record update magnitude
        self.dw = []

    def competition(self, X):
        
        # return the index of best matching neuron
        return np.argmin(np.linalg.norm(self._W - X,axis=1))  

    
    # return the gaussian fall-off neighborhood (or output value y) centering winning neuron  
    def neighborhood(self, X, sigma):
        
        h_x = np.zeros(self.n_features)  ## initialize the neighhood array
        ix = self.competition(X)  ## winning neuron index
        
        for j in range(self.n_features):
            dji = np.abs(j-ix)  ## 1D discrete lattice distance 
            h_x[j] = np.exp(-dji**2/(2*sigma**2))
        
        return h_x
        
    # on step update of the weight vectors 
    def update(self,X,eta,sigma):
        
        h_x = self.neighborhood(X,sigma)  # calculate neighborhood
        dw_length = 0. # record the update magnitude
        
        for j in range(self.n_features):
            
            self._W[j] = self._W[j] + eta*h_x[j]*(X-self._W[j]) 
            
            dw_length = dw_length+ np.linalg.norm(eta*h_x[j]*(X-self._W[j]))
        
        self.dw.append(round(dw_length,6)) # record the update magnitude

    def nodes_center(self,inputdata,sigma=0.5):

        n_input = inputdata.shape[0]

        h = np.zeros((n_input,self.n_features))

        for i in range(n_input):
            h[i] = self.neighborhood(inputdata[i],sigma)

        c = np.zeros((self.n_features,inputdata.shape[1]))

        for j in range(self.n_features):

            c[j] = np.average(inputdata,axis=0,weights = h.T[j])

        return c




class SOM_2D(object):
    
    def __init__(self, nx_features,ny_features, n_input):
        
        self.nx = nx_features  # number of output neurons grid on x axis, 10 here
        
        self.ny = ny_features  # number of output neurons grid on y axis, 10 here
        
        self.n_input = n_input  # number of input dimension
        
        # initialize the synaptic weights, with N(0,1)
        self.W = np.random.normal(0,1,size=(nx_features,ny_features, n_input))
        
        # initialize the list to record update magnitude
        self.dw = []

    def competition(self, X):
        
        # return the 2d index of best matching neuron
        return divmod(np.linalg.norm(self.W - X, axis=2).argmin(),self.ny)

    
    # return the gaussian fall-off neighborhood (or output value y) centering winning neuron  
    def neighborhood(self, X, sigma):
        
        h_x = np.zeros((self.nx,self.ny))  ## initialize the neighhood array
        ix = self.competition(X)  ## winning neuron 2D index
        
        for j in range(self.nx):
            for k in range(self.ny):
                
                dji = np.sqrt( (j-ix[0])**2 + (k-ix[1])**2 )  ## 2D discrete lattice distance 
                
                h_x[j,k] = np.exp(-dji**2/(2*sigma**2)) # gaussian fall-off
        
        return h_x
    
    
    # on step update of the weight vectors 
    def update(self,X,eta,sigma):
        
        h_x = self.neighborhood(X,sigma)  # calculate neighborhood
        dw_length = 0. # record the update magnitude
        
        for j in range(self.nx):
            for k in range(self.ny):
                
                self.W[j,k] = self.W[j,k] + eta*h_x[j,k]*(X-self.W[j,k]) 
                dw_length = dw_length+ np.linalg.norm(eta*h_x[j,k]*(X-self.W[j,k]))
                
        self.dw.append(round(dw_length,6)) # record the update magnitude 


    def nodes_center(self,inputdata,sigma=0.5):
        # initialize the array to store the neighborhood/response
        h_circle = np.zeros((inputdata.shape[0],self.nx,self.ny))

        for i in range(inputdata.shape[0]):
            h[i] = som.neighborhood(inputdata[i],sigma)

        # Then calculate the center of gravity, use weighted average

        ## initialize the center storage arrays
        c = np.zeros((self.nx,self.ny,inputdata.shape[1]))

        for j in range(10):
            for k in range(10):
                c[j,k] = np.average(inputdata, axis=0, weights = h[:,j,k])

        return c




        