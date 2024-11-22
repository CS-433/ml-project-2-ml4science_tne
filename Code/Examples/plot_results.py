import matplotlib.pyplot as plt
from sklearn import metrics
import os
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from Utility import load_results

# General
plt.rcParams["font.family"] = "Calibri"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight = 'bold', titlepad = 20)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight = 'bold', labelpad = 15)   # fontsize of the x and y labels    
plt.rc('xtick', labelsize=SMALL_SIZE, direction = 'out')    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE, direction = 'out')    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes.spines', top=False, right=False)  # Removing the top and right spines   


def plot_results(results_name, title, labels):
    results = load_results(results_name)
    balanced_accuracy_scores = results['balanced_accuracy_scores']
    chance_level_scores = results['chance_level_score']
    mean_chanceLevel = chance_level_scores.mean()
    percentile = np.array([[np.abs(np.percentile(chance_level_scores, 0.5) - mean_chanceLevel)], [np.abs(np.percentile(chance_level_scores, 99.5) - mean_chanceLevel)]])
    
    
    # Plot bar chart with accuracy scores and chance level scores
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusting the figure size for better visibility
    bars1 = ax.bar(
        'Accuracy',
        balanced_accuracy_scores.mean(),
        color='#104E8B',  # Changing the color of the bars
        alpha = 1.0,  # Adding transparency
        zorder=3,  # Increasing the zorder to bring the bars to the front
    )

    # Adding data labels on the bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, 0.05, round(yval, 2), ha='center', va='bottom', fontweight='bold', color='white')
   
    bars2 = ax.bar(
        'Chance level',
        chance_level_scores.mean(),
        yerr=percentile,
        color='#3F6D9B',  # Changing the color of the bars
        alpha = 1.0,  # Adding transparency
        zorder=3,  # Increasing the zorder to bring the bars to the front
        capsize=10,  # Adding cap size for error bars
        error_kw={'elinewidth': 2, 'capthick': 2}
    )

    # Adding data labels on the bars
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, 0.05, round(yval, 2), ha='center', va='bottom', fontweight='bold', color='white')
        
             
    # Grid lines
    ax.yaxis.grid(True, linestyle='-', alpha = 0.7, linewidth=0.5, zorder=0) 

    # Styling the plot
    ax.set_ylabel('Balanced Accuracy')  # Adding label padding
    ax.set_ylim((0, 1))
    ax.set_title(title)  # Increasing title font size and padding
    
    # Save the histogram as png
    plt.savefig(os.path.join(os.getcwd(), os.pardir, "Results", results_name + '_barChart.png'), bbox_inches='tight', dpi=300) 
    plt.show()
    
    # Plot histogram of chance level scores
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(chance_level_scores, label='Chance level', density = True)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Balanced accuracy')
    ax.set_title(title)
    
    # Saving the histogram as png
    plt.savefig(os.path.join(os.getcwd(), os.pardir, "Results", results_name + '_chanceLevel.png'), bbox_inches='tight', dpi=300)  
    plt.show()
    
    # Confusion matrix 
    plt.figure(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay(results['confusion_matrix'], display_labels=labels).plot()
    plt.suptitle(title, fontweight='bold')
    
    # save the confusion matrix as png
    plt.savefig(os.path.join(os.getcwd(), os.pardir, "Results", results_name + '_confusionMatrix.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    return None
