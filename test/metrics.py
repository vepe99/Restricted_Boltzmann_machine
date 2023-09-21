import numpy as np
import itertools



def convert_to_decimal_protein(v):
    '''
    Convert the raw protein list of strings of 1s and 0s 
    to binary numbers
    '''
    protein_list = np.copy(v).astype(str)
    proteins_dataset = np.zeros(protein_list.shape[0]).astype(int)

    for k in np.arange(protein_list.shape[0]):
        proteins_dataset[k] = int(''.join(protein_list[k]),2)
        
    return list(proteins_dataset)

def return_protein_probabilities(decimal_proteins_list, n_amminoacids=5):
    '''
    Returns a pdf. Each element of the returned vector
    represents the probability of each one of the 4^5 proteins
    in the list passed to the function.
    
    The output vector length is always 4^5=1024, independently of
    the input proteins list passed as input.
    '''
    #number of amminoacids in the protein
    n_ammino=n_amminoacids
    
    #total number of preteins, to normalize the pdf
    n_proteins=len(decimal_proteins_list)

    #encoding of amminoacids
    v1,v2,v3,v4=np.eye(4)

    #list all possible combinations of proteins
    possibilities=list(itertools.product(['1000','0100','0010','0001'],
                                         repeat=n_ammino))

    #converts proteins to decimal representation
    decimal=list()
    for p in possibilities:
        decimal.append(int(''.join(p),2))

    #sort the representation from bigger to smaller values
    decimal.sort(reverse=True)

    #initialize a dict with proteins as keys
    protein_cnt=dict.fromkeys(decimal,0)

    #counts the number of times the protein was found
    for protein in decimal_proteins_list:
        protein_cnt[protein]+=1 

    #return the ordered list of 
    #normalized probabilities for each possible protein
    norm_prob=np.array(list(protein_cnt.values()))/n_proteins
    
    return norm_prob

def KL_divergence(p, q):
    pp = np.copy(p)
    pp[pp==0]=1.
    return np.sum(p*np.log(pp/q))

def JS_divergence(p,q):
    m = 0.5*(p+q)
    js = 0.5*(KL_divergence(p, m) + KL_divergence(q, m))
    return js