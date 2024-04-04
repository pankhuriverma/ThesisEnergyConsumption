import subprocess
import time

def get_power_usage_nvidia_smi():
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    power_draw = result.stdout.decode('utf-8').strip()
    return power_draw

start_power = get_power_usage_nvidia_smi()
start_time = time.time()

# Your code here

end_power = get_power_usage_nvidia_smi()
end_time = time.time()
print(start_power)

# Calculate and print energy consumption as before
