import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#function that plots for % autovectors of the energy as a function of the position on the protein  
def distr_ami(v, file_name):
    
    v = (v+1)/2
    
    csi1 = np.array([1,0,0,0])    
    csi2 = np.array([0,1,0,0])   
    csi3 = np.array([0,0,1,0])    
    csi4 = np.array([0,0,0,1])

    #ami_disteach is a matrix, every row is a protein, each entrance of the row will be filled with the autovector assosiated 
    #with the aminoacid in that position. Initially it is set to zeros
    ami_distr = np.zeros((10000,5), dtype=int)

    #assign to every amino acid its autovector as a single number
    #for example if the aminoacid==csi2 than the position of the aminoacid in the ami_distr is filled with a 2
    for i in range(len(v)):
            aminoacids = np.reshape(np.array(v[i, :]), (5,4))
            for a in range(5): 
                if np.array_equal(aminoacids[a], csi1):
                    ami_distr[i,a] = 1
                elif np.array_equal(aminoacids[a], csi2):
                    ami_distr[i,a] = 2
                elif np.array_equal(aminoacids[a], csi3):
                    ami_distr[i,a] = 3
                elif np.array_equal(aminoacids[a], csi4):
                    ami_distr[i,a] = 4

    _, n_ami1 = np.unique(ami_distr[:,0], return_counts=True)
    _, n_ami2 = np.unique(ami_distr[:,1], return_counts=True)
    _, n_ami3 = np.unique(ami_distr[:,2], return_counts=True)
    _, n_ami4 = np.unique(ami_distr[:,3], return_counts=True)
    _, n_ami5 = np.unique(ami_distr[:,4], return_counts=True)

    data = np.vstack((n_ami1, n_ami2, n_ami3, n_ami4,n_ami5))

    df = pd.DataFrame(data/10000, columns=['csi1', 'csi2','csi3','csi4'], index=['1', '2', '3', '4', '5'])
    df.plot(kind='bar')
    plt.savefig(file_name)
