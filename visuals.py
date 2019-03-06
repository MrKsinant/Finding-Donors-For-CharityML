#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#############################################################
# PROGRAMMER: Pierre-Antoine Ksinant                        #
# DATE CREATED: 03/03/2019                                  #
# REVISED DATE: -                                           #
# PURPOSE: This file consists of a library of visualization #
#          functions                                        #
#############################################################


##################
# Needed imports #
##################

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import warnings
from IPython import get_ipython


##########################
# Matplotlib adjustments #
##########################

# Suppress matplotlib user warnings:
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Display inline matplotlib plots with IPython:
get_ipython().run_line_magic("matplotlib", "inline")

# Use retina matplotlib backend with IPython:
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")


#########################
# Function distribution #
#########################

def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure:
    fig = plt.figure(figsize=(11,5))

    # Skewed feature plotting:
    for i, feature in enumerate(["capital-gain", "capital-loss"]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins=25, color="#00A0A0", edgecolor="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.grid(color="black", which="major", axis="y", linestyle="--", linewidth=0.25)

    # Plot aesthetics:
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features",
                     fontsize=16,
                     y =1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features",
                     fontsize=16,
                     y=1.03)
        
    # Display figure:
    fig.tight_layout()
    fig.show()


#####################
# Function evaluate #
#####################

def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners
    """
  
    # Create figure:
    fig, ax = plt.subplots(2, 3, figsize=(11,7))

    # Constants:
    bar_width = 0.3
    colors = ["#33CCFF", "#FFE633", "#FF335B"]
    
    # Super loop to plot four panels of data:
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(["train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"]):
            for i in np.arange(3):
                # Creative plot code:
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width=bar_width, color=colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels:
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles:
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors:
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed")
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed")
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed")
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed")
    
    # Set y-limits for score panels:
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend:
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    plt.legend(handles=patches,
               bbox_to_anchor=(-.80, 2.53),
               loc="upper center",
               borderaxespad=0.,
               ncol=3,
               fontsize="x-large")
    
    # Aesthetics:
    plt.suptitle("Performance Metrics for Three Supervised Learning Models",
                 fontsize=16,
                 y=1.10)
    
    # Display figure:
    plt.tight_layout()
    plt.show()
    

#########################
# Function feature_plot #
#########################

def feature_plot(importances, X_train, y_train):
    """
    Visualization code to display the five most important features
    """
    
    # Display the five most important features:
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create figure:
    fig = plt.figure(figsize=(9,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    plt.bar(np.arange(5),
            values,
            width=0.6,
            align="center",
            color="#33CCFF",
            label="Feature Weight")
    plt.bar(np.arange(5) - 0.3,
            np.cumsum(values),
            width=0.2,
            align="center",
            color="#FFE633",
            label="Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns, rotation=30)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.legend(loc="upper left")
    
    # Display figure:
    plt.tight_layout()
    plt.show()  
