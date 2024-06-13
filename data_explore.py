import numpy as np
import matplotlib.pyplot as plt

small = np.loadtxt("data/CS205_small_Data__25.txt")
large = np.loadtxt("data/CS205_large_Data__3.txt")

class_lables, features = small[:,0], small[:,1:]
class_lables_l, features_l = large[:,0], large[:,1:]

def default_rate(l):
    a = sum(l == l[0]) / len(l)
    return max(a, 1-a)

print(small.shape, large.shape)
print("Default rate: ", default_rate(class_lables))
print("Default rate: ", default_rate(class_lables_l))


min_vals = np.min(features, axis=0)
max_vals = np.max(features, axis=0)
mean_vals = np.mean(features, axis=0)
std_vals = np.std(features, axis=0)

# Number of features (columns)
num_features = features.shape[1]

# Plotting the statistics
fig, ax = plt.subplots()

bar_width = 0.2
opacity = 0.8
index = np.arange(num_features)

bar1 = ax.bar(index - bar_width, min_vals, bar_width, alpha=opacity, color='b', label='Min')
bar2 = ax.bar(index, max_vals, bar_width, alpha=opacity, color='r', label='Max')
bar3 = ax.bar(index + bar_width, mean_vals, bar_width, alpha=opacity, color='g', label='Mean')
bar4 = ax.bar(index + 2*bar_width, std_vals, bar_width, alpha=opacity, color='y', label='Std')

ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Small dataset')
ax.set_xticks(index)
ax.set_xticklabels(index)
ax.legend()

plt.tight_layout()
# plt.show()
plt.savefig("figs/small_table_stat.png")





min_vals = np.min(features_l, axis=0)
max_vals = np.max(features_l, axis=0)
mean_vals = np.mean(features_l, axis=0)
std_vals = np.std(features_l, axis=0)

# Number of features (columns)
num_features = features_l.shape[1]

# Plotting the statistics
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.1
opacity =1
index = np.arange(num_features)

bar1 = ax.bar(index - bar_width, min_vals, bar_width, alpha=opacity, color='b', label='Min')
bar2 = ax.bar(index, max_vals, bar_width, alpha=opacity, color='r', label='Max')
bar3 = ax.bar(index + bar_width, mean_vals, bar_width, alpha=opacity, color='g', label='Mean')
bar4 = ax.bar(index + 2*bar_width, std_vals, bar_width, alpha=opacity, color='y', label='Std')

ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Large dataset')
ax.set_xticks(index)
ax.set_xticklabels(index)
ax.legend()

plt.tight_layout()
# plt.show()
plt.savefig("figs/large_table_stat.png")
