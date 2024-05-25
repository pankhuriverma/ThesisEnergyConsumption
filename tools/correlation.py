import os
import sys
sys.path.append('../models')
sys.path.append('../dataset')

import pandas as pd
import scipy.stats as stats



# Creating a DataFrame from the CSV data
data = pd.read_csv('../dataset/ipc_cycles_dataset/ins_cycles_all_ml_models_dataset.csv')

"""independent_vars = ["L1 Data Cache Misses",
                   "L1 Instruction Cache Misses", "L2 Data Cache Misses", "L2 Instruction Cache Misses",
                   "L1 Cache Misses", "L2 Cache Misses", "L2 Cache Misses",
                   "L3 Load Misses", "Data TLB Misses", "Ins TLB Misses",

                   "L1 Load Misses", "L1 Store Misses", "L2 Load Misses", "L2 Store Misses",

                   "Cycles Stall Waiting for MemWrites", "Cycles With 0 ins issue", "Cycles With max ins issue",
                   "Cycles With no ins comp", "Cycles With max ins comp",

                   "Unconditional branch ins", "Conditional branch ins", "Conditional branch ins taken",
                   "Conditional branch ins not taken", "Conditional branch ins mispred",
                   "Conditional branch ins corr_pred",


                   "Load ins", "Store Ins", "Branch Ins", "Cycle stalled on any resoruce",
                   "Loas/Store ins comp", "L2 data cache access", "L3 data cache access",
                   "L2 data cache reads", "L3 data cache reads", "L2 data cache writes",
                   "L3 data cache writes", "L2 ins cache hits", "L2 ins cache accesses",
                   "L3 ins cache accesses", "L2 ins cache reads", "L3 ins cache reads",
                   "L2 tot cache access", "L3 tot cache access", "L2 tot cache reads",
                   "L3 tot cache reads", "L2 tot cache writes", "L3 tot cache writes",
                   "FLOPS:sing_prec_vec_ops", "FLOPS:dob_prec_vec_ops",
                   "Sing precision vec ins", "Dob prec vec ins", "Ref clock cyc"]"""


independent_vars = ['ins', 'cycles']
# Dependent variables
dependent_vars = ['dram energy', 'cpu energy']

# Calculating Spearman correlation coefficient
correlation_results = {}
for indep_var in independent_vars:
    for dep_var in dependent_vars:
        rho, p_value = stats.spearmanr(data[indep_var], data[dep_var])
        correlation_results[f"{indep_var} and {dep_var}"] = {'rho': rho, 'p_value': p_value}


sorted_data_dict = dict(sorted(correlation_results.items(), key=lambda item: item[1]['rho'],  reverse=True))

print(sorted_data_dict)


with open('../results/correlation_results/correlation_ML_model_ins_cycles.txt', 'w') as f:
    print(sorted_data_dict, file=f)



