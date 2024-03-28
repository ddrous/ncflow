#%%
## Scrape average adaptive time from nohup file

import re

run_folder = "runs/23032024-233458/"
nohup_file = run_folder+"adapt/nohup.log"

# Read nohup file
with open(nohup_file, 'r') as f:
    lines = f.readlines()


# Extract average adaptive time
## A sample line: Gradient descent adaptation time: 0 hours 0 mins 57 secs "
times = []
for line in lines:
    ## Get hours, minutes and seconds
    if "Gradient descent adaptation time:" in line:
        time = re.findall(r'\d+', line)
        times.append(int(time[0])*3600 + int(time[1])*60 + int(time[2]))

## Print times
print(times)

## Calculate average time
avg_time = sum(times)/len(times)
print("Average adapt time: ", avg_time)


#%%
## Training time in seconds : 1 hours 20 mins 48 secs
time = (0*3600 + 5*60 + 36)*1
print(time)


#%%
## Compute mean and std from two numbers
val1 = 0.00567
val2 = 0.00867
mean = (val1 + val2)/2
std = ((val1 - mean)**2 + (val2 - mean)**2)**0.5
print(mean, std)
