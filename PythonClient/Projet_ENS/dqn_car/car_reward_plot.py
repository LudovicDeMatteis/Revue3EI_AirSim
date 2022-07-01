import numpy as np
import matplotlib.pyplot as plt
import os
import re 

path = 'reward_map2'
all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def parse_x(filename):
    result = re.search('x(.*)_', filename)
    return(int(result.group(1)))

all_reward = []
all_x = []

for f in all_files:
    print(f)
    all_x.append(parse_x(f))
    arr = np.load(os.path.join(path, f)).reshape((-1,))
    all_reward.append(arr)

print(np.shape(all_reward))
print(all_x)
reward_sorted = [x for _, x in sorted(zip(all_x, all_reward))]

plt.figure()
plt.imshow(reward_sorted)
plt.xlim([0,80])
plt.show()

print("1")