import numpy as np
import itertools as it
from numpy.polynomial.legendre import leggauss

def gen_ksi_order(nip):
    '''Generates table containing indices for ksi integration points.'''
    
    g = ((j for j in reversed(range(nip))) if i % 2 else (j for j in range(nip)) for i in range(nip))
    return (i for j in g for i in j)

def gen_eta_order(nip):
    '''Generates table containing indices for eta integration points.'''
    
    g = ((j for j in [i]*nip) for i in range(nip))
    return (i for j in g for i in j)

def leg_iterator(order):
    '''order - ksi order or eta order'''
    
    def inner_gen(values):
        '''values are integration points or wages.'''
        
        for i in order:
            yield values[i]
    
    return inner_gen