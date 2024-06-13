import pickle
import glob
import matplotlib.pyplot as plt
import os

for file in glob.glob("outputs/*"):
    print(file)
    with open(file, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)

        data = loaded_dict['meta']
        args = loaded_dict['args']

        print(data)

        plt.figure(figsize=(12, 6))
        # x_labels = ['{' + ','.join(map(str, sorted(fs))) + '}' for fs in data.keys()]
        x_labels = []
        for fs in data.keys():
            sorted_fs = sorted(fs)
            if len(sorted_fs) <= 6:
                x_labels.append('{' + ','.join(map(str, sorted_fs)) + '}')
            else:
                shortened_labels = sorted_fs[:3] + ['...'] + sorted_fs[-3:]
                x_labels.append('{' + ','.join(map(str, shortened_labels)) + '}')
        y_values = list(data.values())
        if "backward" in file:
            y_values[-1] = 0.817 if "large" in args.test_data else 0.769
            x_labels[-1] = 'default rate'

        print(y_values)

        # Show only the first 10 and last 10 elements
        if len(x_labels) > 15:
            x_labels = x_labels[:10] + ['...'] + x_labels[-10:]
            y_values = y_values[:10] + [sum(y_values[10:-10])/len(y_values[10:-10])] + y_values[-10:]


        plt.bar(x_labels, y_values, color='blue')


        # Customize the plot
        plt.xlabel('Feature Sets')
        plt.ylabel('Accuracy')
        plt.title(f'Feature selection: {" ".join(args.algorithm.split("_"))}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Show plot
        # plt.show()
        plt.savefig(f"figs/{os.path.basename(file)}.png")
        # exit()
