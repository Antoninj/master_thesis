# Third-party module imports
import pandas as pd
import pandas_profiling
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import json

# Built-in modules imports
from utils import load_config
config = load_config()


def construct_results_dfs(files):
    """Aggregate all time and frequency feature computations results in dataframes."""

    time_frames = []
    frequency_frames = []
    for filepath in files:
        with open(filepath) as json_data:
            features = json.load(json_data)
            time_features = features["time_features"]
            frequency_features = features["frequency_features"]

        time_frames.append(pd.DataFrame(time_features, index=[0]))
        frequency_frames.append(pd.DataFrame(frequency_features, index=[0]))

    time_features_df = pd.concat(time_frames, axis=0)
    frequency_features_df = pd.concat(frequency_frames, axis=0)

    df1 = time_features_df.reset_index().drop('index', 1)
    df2 = frequency_features_df.reset_index().drop('index', 1)

    return [df1, df2]


def generate_profile_report(df, filename, bins=50):
    """
    Create a HTML profile report of a dataframe using general descriptive statistics.

    The profile report is generated using the pandas profiling (https://github.com/pandas-profiling) library.
    """

    # Create the report
    df_profile = pandas_profiling.ProfileReport(df, bins=bins)

    # Save the report
    df_profile.to_file(outputfile=filename)


def generate_all_profile_reports(wbb_dataframes, fp_dataframes, statistics_results_folder):
    """Create all the profile reports."""

    domain_names = ["time_domain_features", "frequency_domain_features"]
    wbb_report_names = ["{}/wbb_{}_report.html".format(statistics_results_folder, name) for name in domain_names]
    fp_report_names = ["{}/fp_{}_report.html".format(statistics_results_folder, name) for name in domain_names]

    dfs = wbb_dataframes + fp_dataframes
    report_names = wbb_report_names + fp_report_names
    for (data, name) in zip(dfs, report_names):
        generate_profile_report(data, name)


def compute_mean_and_stds(df1, df2):
    """Compute the mean and standard deviation values for each feature."""

    wbb_mean_df = pd.DataFrame(df1.mean(), columns=["WBB mean"])
    wbb_std_df = pd.DataFrame(df1.std(), columns=["WBB std"])
    fp_mean_df = pd.DataFrame(df2.mean(), columns=["FP mean"])
    fp_std_df = pd.DataFrame(df2.std(), columns=["FP std"])

    aggregated_results = pd.concat([wbb_mean_df, wbb_std_df, fp_mean_df, fp_std_df], axis=1)

    return aggregated_results


def compute_spearman_correlation(df1, df2):
    """                     ."""

    result_dict = {}
    # Loop over each feature
    for column in df1.columns:
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Perform the T-test
        rho, p_value = stats.spearmanr(x, y, nan_policy="propagate")

        # Store the results
        result_dict[column] = {}
        result_dict[column]["rho"] = rho
        result_dict[column]["p-value"] = p_value

    return result_dict


def compute_t_test(df1, df2):
    """                     ."""

    result_dict = {}
    # Loop over each feature
    for column in df1.columns:
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Perform the T-test
        t_statistic, p_value = stats.ttest_ind(x, y, nan_policy="propagate")

        # Store the results
        result_dict[column] = {}
        result_dict[column]["t_statistic"] = t_statistic
        result_dict[column]["p-value"] = p_value

    return result_dict


def pearson_correlation_plots(df1, df2, statistics_results_folder, name="time_domain_features"):
    """Perform a linear least-squares regression and plot the correlation line for each feature."""

    fig, axs = plt.subplots(8, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs[-1, -1].axis('off')

    result_dict = {}
    # Loop over each feature
    for ax, column in zip(axs.ravel(), df1.columns):
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Perform the linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Store the linear regression results
        result_dict[column] = {}
        result_dict[column]["slope"] = slope
        result_dict[column]["intercept"] = intercept
        result_dict[column]["R"] = r_value
        result_dict[column]["p-value"] = p_value

        # Make the plot
        ax.plot(x, y, '.', label='original data')
        ax.plot(x, intercept + slope * x, 'black', label='fitted line', linewidth=0.3)
        ax.set_xlabel('Balance Board')
        ax.set_ylabel('Force plate')
        ax.set_title(column, weight=600)
        ax.text(0.8, 0.9, "p-value = {}".format(round(p_value, 4)), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8, 0.8, "R\u00b2={}".format(round(r_value**2, 4)), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8, 0.7, "Slope = {}".format(round(slope, 4)), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8, 0.6, "Intercept = {}".format(round(intercept, 4)), fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # ax.legend()

    # Save the plots
    plt.savefig("{}/{}_correlation_plots.png".format(statistics_results_folder, name), bbox_inches='tight')

    return result_dict


def bland_altman_plots(df1, df2, statistics_results_folder, name="time_domain"):

    fig, axs = plt.subplots(8, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    axs[-1, -1].axis('off')

    result_dict = {}
    # Loop over each feature
    for ax, column in zip(axs.ravel(), df1.columns):
        x = df1[column]
        y = df2[column][:df1.shape[0]]

        # Prepare the data for the plots
        mean = np.mean([x, y], axis=0)
        diff = x - y
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        result_dict[column] = {}
        result_dict[column]["LOA"] = "{},{}".format(md - 2 * sd, md + 2 * sd)

        # Make the plot
        ax.scatter(mean, diff, marker='.', s=100, color="gray")
        ax.axhline(md, color='tomato', linestyle='--')
        ax.axhline(md + 2 * sd, color='teal', linestyle='--')
        ax.axhline(md - 2 * sd, color='teal', linestyle='--')
        ax.set_xlabel('Mean of the two systems')
        ax.set_ylabel('Mean of the difference')
        ax.set_title(column, weight=600)

        # ax.legend()

    # Save the plots
    plt.savefig("{}/{}_bland_altman_plots.png".format(statistics_results_folder, name), bbox_inches='tight')

    return result_dict
