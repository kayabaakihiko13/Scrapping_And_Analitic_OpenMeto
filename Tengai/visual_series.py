import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from typing import Any,Union

def lineplot_features(data:Union[pd.Series,pd.DataFrame],
                      size:tuple[int]=(12,6),title:str=None) -> Any:
    """
    this function for visual line one line feature
    Args:
        data (pd.Series): input data
    """
    if isinstance(data,pd.Series):
        title = f"{data.name}" if title is None else title
        # setting size plot
        _ = plt.figure(figsize=size)
        sns.lineplot(data=data,label=f"line of {data.name}")
        plt.title(title)
        plt.ylabel("Values")
        plt.legend(loc="lower right")
        plt.show()
    if isinstance(data,pd.DataFrame):
        if data.shape[1] >1:
            title = f"{data.columns[0]}" if title is None else title
            # setting size plot
            n_col = len(data.columns)
            n_row = min(data.shape[1],1)
            
            _, ax = plt.subplots(n_row,n_col,figsize=size)
            
            for i,feat in enumerate(data.columns):
                sns.lineplot(data[feat],ax=ax[i])
                ax[i].set_xlabel(f"{feat}")
            plt.suptitle("Line Plot feature dataFrames")
            plt.show()
        else:
            data = pd.Series(data=data.values[:, 0],
                             name=data.columns[0])
            lineplot_features(data)


def lineplot_resample_feature(data: pd.DataFrame, on_feat: str, rule: str) -> None:
    """_summary_

    Args:
        data (pd.DataFrame): input data
        on_feat (str): input on_feature sample order by feature
        rule (str): aturan
    """
    # Resample data
    resample_data = data.resample(rule, on=on_feat).sum()
    if resample_data.shape[1] > 1:
    # Calculate the number of features
        n_features = len(data.columns) - 1  # Subtract 1 for 'on_feat'
        n_col = min(n_features, 2)
        n_row = n_features // (n_col)

        # Setting up the figure for plotting
        fig, ax = plt.subplots(n_row + 1, n_col, figsize=(15, 6))

        for i, feat in enumerate(data.columns.drop(on_feat)):
            sns.lineplot(x=resample_data.index, y=resample_data[feat], ax=ax[i // 2, i % 2])
            ax[i // 2, i % 2].set_xlabel(on_feat)
            ax[i // 2, i % 2].set_ylabel(f"{feat}")

        # deactivate plot no useage
        for i in range(n_features, (n_row + 1) * n_col):
            fig.delaxes(ax[i // n_col, i % n_col])

        # Adjust the layout and title
        plt.tight_layout()
        plt.suptitle("Line Plot Resampled Dataframes", fontsize=15)
        plt.show()
    else:
        resample_data = pd.Series(data=resample_data.values[:, 0], name=resample_data.columns[0])
        lineplot_features(resample_data)
      


def average_generating_barplot(data: pd.DataFrame, on_feat: str, size: tuple[int] = (30, 27)):
    """
    Generate a bar plot of the mean values of numeric features grouped by the hour component of on_feat.
    Highlight the bar with the maximum value in red.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        on_feat (str): The feature used for grouping and x-axis labels.
        size (tuple[int]): The size of the plot (width, height).
    """
    # Filter numeric columns
    numeric_feat = data.select_dtypes(include=np.number).columns[1:]
    
    # Calculate the mean values and reset the index
    mean_data = data.groupby(data[on_feat].dt.hour)[numeric_feat].mean().reset_index()
    _, ax = plt.subplots(ncols=2, nrows=3, figsize=size)
    colors = sns.color_palette("muted")
    for i, column in enumerate(numeric_feat):
        sns.barplot(x=on_feat, y=column, data=mean_data, ax=ax[i//2, i%2], color=colors[i%len(colors)])
        ax[i//2, i%2].set_title("{} Avarange generating time".format(column), fontsize=22)
        ax[i//2, i%2].set_xlabel("Time")
        ax[i//2, i%2].set_ylabel("Mean Value")
        ax[i//2, i%2].spines["top"].set_visible(False)
        ax[i//2, i%2].spines["right"].set_visible(False)
        ax[i//2, i%2].tick_params(axis="x", labelrotation=45)

        # Find maximum value and its index
        max_val = mean_data[column].max()
        max_idx = mean_data[mean_data[column] == max_val][on_feat].iloc[0]

        # Highlight and annotate the bar with maximum value
        rects = ax[i//2, i%2].containers[0]
        for rect in rects:
            if rect.get_height() == max_val:
                rect.set_color("red")
                ax[i//2, i%2].annotate(f"{max_val:.2f}", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), 
                                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    plt.show()


def plot_rolling_statistics(df: pd.DataFrame, rolling_window:int=7)->Any:
    """
    Plot rolling statistics for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        rolling_window (int, optional): Rolling window size for mean and std calculations. Defaults to 7.

    Returns:
        None: This function displays the plot but doesn't return any value.
    """
    numerical_features = df.select_dtypes(include=np.number).columns
    f, ax = plt.subplots(nrows=len(numerical_features), ncols=1, figsize=(20, 12))
    
    for i, feat in enumerate(numerical_features):
        sns.lineplot(x=df['time'], y=df[feat], ax=ax[i], color='dodgerblue')
        sns.lineplot(x=df['time'], y=df[feat].rolling(rolling_window).mean(), ax=ax[i], color='black', label='rolling mean')
        sns.lineplot(x=df['time'], y=df[feat].rolling(rolling_window).std(), ax=ax[i], color='orange', label='rolling std')
        
        # Calculate ADF test results
        adf_result = adfuller(df[feat], autolag="AIC")
        if adf_result[1] <= 0.05:
            ax[i].set_title(f'{feat}: the time series is stationary', fontsize=14)
        else:
            ax[i].set_title(f'{feat}: the time series is non-stationary', fontsize=14)

        # Add labels and legend
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel('Drainage Volume')
        ax[i].legend()

    # Show the plot
    plt.show()
    
def visualize_adfuller_results(data: pd.DataFrame, feature: Union[str, List[str]], title: str) -> None:
    if isinstance(feature, str):
        result = adfuller(data[feature].values)
        feature_name = feature  # Store the feature name for the title
        visualize_adfuller(data, result, title, feature_name)
    elif isinstance(feature, list):
        for feat in feature:
            result = adfuller(data[feat].values)
            visualize_adfuller(data, result, title, feat)

def visualize_adfuller(data: pd.DataFrame, result: tuple[np.number], title: str, feature_name: str) -> Any:
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if p_val < significance_level and adf_stat < crit_val_1:
        linecolor = 'forestgreen'
    elif p_val < significance_level and adf_stat < crit_val_5:
        linecolor = 'orange'
    elif p_val < significance_level and adf_stat < crit_val_10:
        linecolor = 'red'
    else:
        linecolor = 'purple'

    plt.figure(figsize=(12, 10))
    sns.lineplot(x=data.index, y=data[feature_name], color=linecolor)  # Assuming 'time' is the DataFrame index
    plt.title(f'{feature_name}\nADF Statistic {adf_stat:0.3f}, p-value: {p_val}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    plt.ylabel(ylabel=title, fontsize=14)
    plt.show()  # Show the plot
