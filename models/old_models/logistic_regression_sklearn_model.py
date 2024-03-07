# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pypapi import papi_high, events as papi_events
import pyRAPL
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time
pyRAPL.setup()

def measure_model_perf_energy(event_name, X_subset, X, y):

        meter = pyRAPL.Measurement('LR Model')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        meter.begin()
        start_time = time.time()
        papi_high.start_counters([event_name])
        logreg = LogisticRegression(max_iter=100)
        logreg.fit(X_train, y_train)
        counters = papi_high.stop_counters()
        end_time = time.time()
        meter.end()
        tot_time = end_time - start_time
        output = meter.result
        cpu_energy = output.pkg[0]  # Assuming single-socket CPU; adjust as necessary
        dram_energy = output.dram[0]
        total_energy = cpu_energy + dram_energy

        return counters[0], cpu_energy, dram_energy, total_energy, tot_time

if __name__ == "__main__":
    event_data = []
    total_cycles = []
    total_energy = []
    cpu_energy = []
    dram_energy = []
    total_time = []
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





    energy_names = ["cpu_energy", "dram_energy", "total_energy", "total time"]

    """diabetes = datasets.load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target"""

    """# Load the Boston housing dataset
    housing = datasets.fetch_california_housing()
    X = housing.data  # Features
    y = housing.target"""

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable

    counter = 0
    for event, event_name in zip(events, event_names):

         for i in range(0, 5):  # 10 features in total, indexed from 1 to 10 for readability
                # Use the first i features
                X_subset = X[:,:i]
                print(X_subset)

                perf_result, cpu_ene, dram_ene, total_ene, tot_time = measure_model_perf_energy(event, X_subset, X, y)
                event_data.append(perf_result)

                if counter >= 1:
                    continue

                cpu_energy.append(cpu_ene)
                dram_energy.append(dram_ene)
                total_energy.append(total_ene)
                total_time.append(tot_time)

         counter = counter + 1
         all_events[event_name] = event_data
         event_data = []

    all_events["cpu energy"] = cpu_energy
    all_events["dram energy"] = dram_energy
    all_events["total energy"] = total_energy
    all_events["total time"] = total_time



    df = pd.DataFrame(all_events)
    csv_file = 'linear_model_sklearn_diabetes_perf_dataset.csv'  # Specify your CSV file name
    df.to_csv(csv_file, index=False, mode = 'a')

