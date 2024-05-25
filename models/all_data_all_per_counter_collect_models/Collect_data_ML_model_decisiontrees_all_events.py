# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pypapi import papi_high, events as papi_events
import pyRAPL
import pandas as pd
import time
from pypapi import papi_high, events as papi_events
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
pyRAPL.setup()

def measure_model_perf_energy(event_name, X_train, y_train):

    sum_counter = 0
    sum_cpu_energy = 0
    sum_dram_energy = 0
    cnt = 0
    for index in range(5):
        meter = pyRAPL.Measurement('LR Model')
        meter.begin()

        papi_high.start_counters([event_name])
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(X_train, y_train)
        perf_counter_value = papi_high.stop_counters()
        meter.end()
        output = meter.result
        cpu_ener = output.pkg[0] / 1000000 # Assuming single-socket CPU; adjust as necessary
        dram_ener = output.dram[0] / 1000000
        sum_counter += perf_counter_value[0]
        sum_cpu_energy += cpu_ener
        sum_dram_energy += dram_ener
        cnt += cnt


    # Calculate the averages of the energy measurements

    avg_counter = sum_counter / 5
    avg_cpu_energy = sum_cpu_energy / 5
    avg_dram_energy = sum_dram_energy / 5



    return avg_counter, avg_cpu_energy, avg_dram_energy

if __name__ == "__main__":
    event_data = []
    cpu_energy = []
    dram_energy = []
    counter = []
    all_events = {}

    events = [papi_events.PAPI_TOT_INS, papi_events.PAPI_TOT_CYC, papi_events.PAPI_L1_DCM,
              papi_events.PAPI_L1_ICM, papi_events.PAPI_L2_DCM, papi_events.PAPI_L2_ICM,
              papi_events.PAPI_L1_TCM, papi_events.PAPI_L2_TCM, papi_events.PAPI_L3_TCM,
              papi_events.PAPI_L3_LDM, papi_events.PAPI_TLB_DM,
              papi_events.PAPI_TLB_IM, papi_events.PAPI_L1_LDM,
              papi_events.PAPI_L1_STM, papi_events.PAPI_L2_LDM, papi_events.PAPI_L2_STM,

              papi_events.PAPI_MEM_WCY, papi_events.PAPI_STL_ICY, papi_events.PAPI_FUL_ICY,
              papi_events.PAPI_STL_CCY, papi_events.PAPI_FUL_CCY,

              papi_events.PAPI_BR_UCN, papi_events.PAPI_BR_CN, papi_events.PAPI_BR_TKN,
              papi_events.PAPI_BR_NTK, papi_events.PAPI_BR_MSP, papi_events.PAPI_BR_PRC,

              papi_events.PAPI_LD_INS, papi_events.PAPI_SR_INS, papi_events.PAPI_BR_INS,
              papi_events.PAPI_RES_STL, papi_events.PAPI_LST_INS, papi_events.PAPI_L2_DCA,
              papi_events.PAPI_L3_DCA, papi_events.PAPI_L2_DCR, papi_events.PAPI_L3_DCR,
              papi_events.PAPI_L2_DCW, papi_events.PAPI_L3_DCW, papi_events.PAPI_L2_ICH,
              papi_events.PAPI_L2_ICA, papi_events.PAPI_L3_ICA, papi_events.PAPI_L2_ICR,
              papi_events.PAPI_L3_ICR, papi_events.PAPI_L2_TCA, papi_events.PAPI_L3_TCA,
              papi_events.PAPI_L2_TCR, papi_events.PAPI_L3_TCR, papi_events.PAPI_L2_TCW,
              papi_events.PAPI_L3_TCW, papi_events.PAPI_SP_OPS, papi_events.PAPI_DP_OPS,
              papi_events.PAPI_VEC_SP, papi_events.PAPI_VEC_DP, papi_events.PAPI_REF_CYC]

    event_names = ["Total Instructions", "Total Cycles", "L1 Data Cache Misses",
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
                   "Sing precision vec ins", "Dob prec vec ins", "Ref clock cyc"]

    # Fetch the California housing dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    print(X.shape)
    print(y.shape)

    cnt = 0
    print("length of x:",  len(X))

    counter = 0
    counter2 = 0
    for event, event_name in zip(events, event_names):
        for i in range(100, 455):  # X.shape[1] gives the number of columns (features)
                    X_data = X[:i, :]
                    y_data = y[:i]

                    # Split the data into training/testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

                    event_value, cpu_ene, dram_ene = measure_model_perf_energy(event, X_train, y_train)
                    event_data.append(event_value)
                    if counter >= 1:
                        continue

                    cpu_energy.append(cpu_ene)
                    dram_energy.append(dram_ene)

        counter = counter + 1
        all_events[event_name] = event_data
        event_data = []
        print(counter)

    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy

    df = pd.DataFrame(all_events)
    csv_file = '../../dataset/all_pmc_dataset/ML_model_decisiontree_dataset.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode='a')

