import numpy as np
import helper_functions
from PyMoments import kstat
from itertools import product

# function to calculate the second and fourth cumulant tensors from observed data
def cumulant_tensors(observeddata):
    I=observeddata.shape[1]
    sym_indices,b,c=helper_functions.symmetric_indices(I,4)
    fourth_cumulants=np.apply_along_axis(lambda x: kstat(observeddata,tuple(x)), 0, sym_indices)
    fourth_cumulant_dict={tuple(sym_indices[:,n]):fourth_cumulants[n] for n in range(len(fourth_cumulants))}
    all_indices=np.array([list(i) for i in product(range(I), range(I),range(I),range(I))])
    values=np.apply_along_axis(lambda x:fourth_cumulant_dict[tuple(np.sort(x))],1,all_indices)
    fourth_order_kstats=values.reshape(I,I,I,I)
    
    sym_indices_2,b,c=helper_functions.symmetric_indices(I,2)
    second_cumulants=np.apply_along_axis(lambda x: kstat(observeddata,tuple(x)), 0, sym_indices_2)
    second_cumulant_dict={tuple(sym_indices_2[:,n]):second_cumulants[n] for n in range(len(second_cumulants))}
    all_indices_2=np.array([list(i) for i in product(range(I), range(I))])
    values=np.apply_along_axis(lambda x:second_cumulant_dict[tuple(np.sort(x))],1,all_indices_2)
    second_order_kstats=values.reshape(I,I)
    return second_order_kstats,fourth_order_kstats