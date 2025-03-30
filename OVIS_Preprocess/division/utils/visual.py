#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch


def draw_bar(x_values, y_values, title, x_label, y_label):
    """
    Draw bar chart
    :param x_values: x-axis values
    :param y_values: y-axis values
    :param title: title of the bar chart
    :param x_label: x-axis label
    :param y_label: y-axis labels
    """
    # direct assignment creates a DataFrame
    data = pd.DataFrame({'Categories': x_values, 'Values': y_values})

    # set the color mapping
    colors = sns.color_palette("Spectral", n_colors=len(x_values))
    # camp = plt.get_cmap("Spectral")

    plt.figure(dpi=300)
    # Create Bar Charts
    ax = sns.barplot(x='Categories', y='Values', hue='Categories', data=data, palette=colors, legend=False)

    # get current scale range
    locs = range(len(x_values))

    # set the fixed position
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))

    # set the axis labels
    ax.set_xticklabels(labels=x_values, rotation=60, fontsize=8)

    # setup chart titles and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    # Show charts
    plt.show()


def draw_stacked_bar(x_values, y_values1, y_values2, title, x_label, y_label):
    """
    Draw stacked bar chart
    :param x_values: x-axis values
    :param y_values1: y-axis values1
    :param y_values2: y-axis values2
    :param title: title of the bar chart
    :param x_label: x-axis label
    :param y_label: y-axis labels
    """
    # direct assignment creates DataFrame
    data1 = pd.DataFrame({'Categories': x_values, 'Values': y_values1})
    data2 = pd.DataFrame({'Categories': x_values, 'Values': y_values2})

    # creat subplots
    fig, ax = plt.subplots(dpi=300)

    # Create Bar Charts
    sns.barplot(x='Categories', y='Values', ax=ax, data=data1, color='pink')
    sns.barplot(x='Categories', y='Values', ax=ax, data=data2, color='skyblue', bottom=y_values1)

    # get current scale range
    locs = range(len(x_values))

    # set the fixed position
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))

    # set the axis labels
    ax.set_xticklabels(labels=x_values, rotation=60, fontsize=8)

    # setup chart titles and labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # setup legend
    legend_elements = [Patch(facecolor='skyblue', edgecolor='skyblue', label='Training Set'),
                       Patch(facecolor='pink', edgecolor='pink', label='Validation Set')]

    ax.legend(handles=legend_elements, title='Dataset', loc='upper right')

    fig.tight_layout()

    # Show charts
    plt.show()
