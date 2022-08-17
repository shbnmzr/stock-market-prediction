import matplotlib.pyplot as plt
import seaborn as sns
from financial_functions import *
sns.color_palette("husl", 8)


def plot_feature(data, feature):
    plt.figure(figsize=(16, 8))
    line_plot = sns.lineplot(x=data, y=data[feature])
    line_plot.set_title(f'{feature}', fontdict={'fontsize':16}, pad=16)

    plt.show()


def plot_correlations(data):
    plt.figure(figsize=(16, 8))
    mask = np.triu(np.ones_like(data.corr(), dtype=bool))
    heatmap = sns.heatmap(data.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 16}, pad=16)

    plt.show()


def plot_moving_average(data, duration='monthly'):
    data = compute_moving_average(data, duration)
    plt.figure(figsize=(16, 8))
    sns.lineplot(x=data.index, y=data['Close'])

    plt.show()
