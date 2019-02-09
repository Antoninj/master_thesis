# Third-party module imports
import json
import logging

import numpy as np
import pandas as pd
import pandas_profiling
from matplotlib import pyplot as plt
from rpy2.robjects import DataFrame, FloatVector, pandas2ri
from rpy2.robjects.packages import importr
from scipy import stats
# Built-in modules imports
from utils import load_config, setup_logging

config = load_config()

setup_logging()
logger = logging.getLogger("stats")


def create_index(df, file_info_items):
    """Create a new multi level index."""

    arrays = [df["device"].values, df["subject"].values,
              df["trial"].values, df["balance board"].values]
    tuples = list(zip(*arrays))
    return pd.MultiIndex.from_tuples(tuples, names=file_info_items)


def construct_results_dfs(files):
    """Aggregate all time and frequency feature computations results in two dataframes."""

    time_frames = []
    frequency_frames = []
    file_info_items = ['device', 'subject', 'trial', 'balance board']

    # Create lists of dataframes with all the time and frequency feature computations results
    for filepath in files:
        with open(filepath) as json_data:
            features = json.load(json_data)
            time_features = features["time_features"]
            frequency_features = features["frequency_features"]
            for dic in (time_features, frequency_features):
                dic["device"] = features["device"]
                dic["subject"] = features["subject"]
                dic["trial"] = features["trial"]
                dic["balance board"] = features["balance board"]

        time_frames.append(pd.DataFrame(time_features, index=[0]))
        frequency_frames.append(pd.DataFrame(frequency_features, index=[0]))

    # Concatenate the dataframes together
    time_features_df = pd.concat(time_frames, axis=0)
    frequency_features_df = pd.concat(frequency_frames, axis=0)

    # Reshape the dataframes
    df1 = time_features_df.reset_index().drop(file_info_items, 1)
    df1.index = create_index(time_features_df, file_info_items)
    df1.drop('index', 1, inplace=True)
    df1.sort_index(inplace=True)

    df2 = frequency_features_df.reset_index().drop(file_info_items, 1)
    df2.index = create_index(frequency_features_df, file_info_items)
    df2.drop('index', 1, inplace=True)
    df2.sort_index(inplace=True)

    return [df1, df2]


def generate_profile_report(df, filename, bins=50):
    """
    Create a HTML profile report of a dataframe using general descriptive statistics.

    Notes
    -----
    The profile report is generated using the pandas profiling (https://github.com/pandas-profiling) library.
    """

    # Create the profile report
    df_profile = pandas_profiling.ProfileReport(df, bins=bins, check_correlation=False)

    # Save the report
    df_profile.to_file(outputfile=filename)


def generate_all_profile_reports(dataframes, statistics_results_folder):
    """Create all the profile reports."""

    wbb_numbers = ["1", "2", "3", "4"]
    html_report_names = ["{}/{}_report.html".format(statistics_results_folder, number) for number in wbb_numbers]

    dfs = [dataframes.loc[(dataframes.index.get_level_values(3) == number)] for number in wbb_numbers]
    for (data, name) in zip(dfs, html_report_names):
        generate_profile_report(data, name)


def compute_mean_and_stds(df1, df2, statistics_results_folder):
    """Compute the mean and standard deviation values for each feature."""

    wbb_and_fp_results = pd.concat([df1, df2], axis=0)

    # Group by WBB and compute the mean value for each feature
    feature_mean_results = wbb_and_fp_results.groupby(
        [wbb_and_fp_results.index.get_level_values(0), wbb_and_fp_results.index.get_level_values(3)]).mean().transpose().stack(0).unstack()

    # Group by WBB and compute the mean value for each feature
    feature_std_results = wbb_and_fp_results.groupby(
        [wbb_and_fp_results.index.get_level_values(0), wbb_and_fp_results.index.get_level_values(3)]).std().transpose().stack(0).unstack()

    aggregated_results = (feature_mean_results, feature_std_results)

    # Save the results
    mean_report_name = "{}/mean_results.csv".format(statistics_results_folder)
    std_report_name = "{}/stds_results.csv".format(statistics_results_folder)
    aggregated_results[0].to_csv(mean_report_name, sep=',', encoding='utf-8')
    aggregated_results[1].to_csv(std_report_name, sep=',', encoding='utf-8')

    return aggregated_results


def compute_spearman_correlation(df1, df2, statistics_results_folder):
    """
    Compute the spearman correlation coefficient between the WBB and Force plate data for each feature.

    References
    ----------
    .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    """

    wbb_numbers = ["1", "2", "3", "4"]
    dfs_1 = [df1.loc[(df1.index.get_level_values(3) == number)] for number in wbb_numbers]
    dfs_2 = [df2.loc[(df2.index.get_level_values(3) == number)] for number in wbb_numbers]

    result_dict = {key:{} for key in df1.columns}

    # Loop over each WBB data
    for (df1, df2, number) in zip(dfs_1, dfs_2, wbb_numbers ):
        # Loop over each feature
        for column in df1.columns:
            x = df1[column]
            y = df2[column]

            try:
                # Compute the spearman coefficient(rho) and the corresponding p-value
                rho, p_value = stats.spearmanr(x, y, nan_policy="propagate")

                # Store the results
                result_dict[column][number] = {}
                result_dict[column][number]["p-value"] = p_value
                result_dict[column][number]["rho"] = rho

            except (RuntimeWarning, Exception) as err:
                logger.error("Problem with feature: {}.\n{}".format(column, err), exc_info=True, stack_info=True)
                pass

    # Reshape the raw results
    result_dict_collapsed = {(outer_k, inner_k): inner_v for outer_k in result_dict
                             for inner_k, inner_v in result_dict[outer_k].items()}

    aggregated_results = pd.DataFrame.from_dict(result_dict_collapsed).transpose()
    aggregated_results = aggregated_results.unstack().stack(0).unstack()

    # Save the results
    report_name = "{}/spearman_correlation_results.csv".format(statistics_results_folder)
    aggregated_results.to_csv(report_name, sep=',', encoding='utf-8')

    return aggregated_results


def perform_t_test(df1, df2, statistics_results_folder):
    """"
    Perfom a T-test using the WBB and Force plate data for each feature.

    References
    ----------
    .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
    """

    result_dict = {}
    # Loop over each feature
    for column in df1.columns:
        x = df1[column]
        y = df2[column]

        try:
            # Compute the T-statistic and the corresponding p-value
            t_statistic, p_value = stats.ttest_ind(x, y, nan_policy="propagate")

            # Store the results
            result_dict[column] = {}
            result_dict[column]["t_statistic"] = t_statistic
            result_dict[column]["p-value"] = p_value

        except (RuntimeWarning, Exception) as err:
            logger.error("Problem with feature: {}.\n{}".format(column, err), exc_info=True, stack_info=True)
            pass

    # Save the results
    result_dict_df = pd.DataFrame.from_dict(result_dict).transpose()
    report_name = "{}/t_test.csv".format(statistics_results_folder)
    result_dict_df.to_csv(report_name, sep=',', encoding='utf-8')

    return result_dict


def make_pearson_correlation_plots(df1, df2, statistics_results_folder, plot_size):
    """
    Perform a linear least-squares regression and plot the correlation line for each feature.

    References
    ----------
    .. [1] Scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
    """

    wbb_numbers = ["1", "2", "3", "4"]
    dfs_1 = [df1.loc[(df1.index.get_level_values(3) == number)] for number in wbb_numbers]
    dfs_2 = [df2.loc[(df2.index.get_level_values(3) == number)] for number in wbb_numbers]

    result_dict = {key:{} for key in df1.columns}

    # Loop over each WBB data
    for (df1, df2, number) in zip(dfs_1, dfs_2, wbb_numbers):

        fig, axs = plt.subplots(plot_size, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5)
        #axs[-1, -1].axis('off')

        # Loop over each feature
        for ax, column in zip(axs.ravel(), df1.columns):
            x = df1[column]
            y = df2[column]

            try:
                # Perform the linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Store the linear regression results
                result_dict[column][number] = {}
                result_dict[column][number]["slope"] = round(slope, 4)
                result_dict[column][number]["intercept"] = round(intercept, 4)
                result_dict[column][number]["R"] = round(r_value, 4)
                result_dict[column][number]["p-value"] = round(p_value, 4)

                # Make the plot
                ax.plot(x, y, '.', label='original data')
                ax.plot(x, intercept + slope * x, 'black', label='fitted line', linewidth=0.3)
                ax.set_xlabel('Balance Board')
                ax.set_ylabel('Force plate')
                ax.set_title(column, weight=600)
                ax.text(0.8, 0.4, "p-value = {}".format(round(p_value, 4)), fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                ax.text(0.8, 0.3, "R\u00b2={}".format(round(r_value**2, 4)), fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                ax.text(0.8, 0.2, "Slope = {}".format(round(slope, 4)), fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                ax.text(0.8, 0.1, "Intercept = {}".format(round(intercept, 4)), fontsize=9, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes)
                # ax.legend()

            except (RuntimeWarning, Exception) as err:
                logger.error("Problem with feature: {}.\n{}".format(column, err), exc_info=True, stack_info=True)
                pass

        # Save the plots
        plt.savefig("{}/balance_board_{}_linear_regression_plots.png".format(statistics_results_folder, number), bbox_inches='tight')

    # Reshape the raw results
    result_dict_collapsed = {(outer_k, inner_k): inner_v for outer_k in result_dict
                                 for inner_k, inner_v in result_dict[outer_k].items()}
    aggregated_results = pd.DataFrame.from_dict(result_dict_collapsed).transpose()
    aggregated_results = aggregated_results.unstack().stack(0).unstack()

    # Save the results
    report_name = "{}/linear_regression_results.csv".format(statistics_results_folder)
    aggregated_results.to_csv(report_name, sep=',', encoding='utf-8')

    return result_dict


def make_bland_altman_plots(df1, df2, statistics_results_folder, plot_size):
    """Compute limit of agreement values and make bland and altman plot for each feature."""

    fig, axs = plt.subplots(plot_size, 3, figsize=(20, 30), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5)
    df1 = df1.reorder_levels(['balance board', 'device', 'subject', 'trial']).sort_index()
    df2 = df2.reorder_levels(['balance board', 'device', 'subject', 'trial']).sort_index()

    result_dict = {}
    # Loop over each feature
    for ax, column in zip(axs.ravel(), df1.columns):
        x = df1[column].values
        y = df2[column].values

        try:
            # Compute the LOA and arrange the data for the plots
            #mean = np.mean([x, y], axis=0)
            trials = [trial for trial in range(len(x))]
            diff = x - y
            md = np.mean(diff)
            sd = np.std(diff, axis=0)

            # Store the results
            result_dict[column] = {}
            result_dict[column]["LOA"] = "{},{}".format(md - 2 * sd, md + 2 * sd)

            # Make the plot
            ax.scatter(trials, diff, marker='.', s=60, color="gray", linewidth=0.5)
            trials_limits = [x * 18 + 0.5 for x in range(1, 4)]
            for limit in trials_limits:
                ax.axvline(limit, color='black', linestyle='--', linewidth=0.5)
            ax.axhline(md, color='tomato', linestyle='--')
            ax.axhline(md + 2 * sd, color='teal', linestyle='--', linewidth=0.5)
            ax.axhline(md - 2 * sd, color='teal', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Trials')
            ax.set_ylabel('Difference')
            ax.set_title(column, weight=600)

            # ax.legend()

        except (RuntimeWarning, Exception) as err:
            logger.error("Problem with feature: {}.\n{}".format(column, err), exc_info=True, stack_info=True)
            pass

    # Save the plots
    plt.savefig("{}/bland_altman_plots.png".format(statistics_results_folder),  bbox_inches='tight')

    # Save the results
    result_dict_df = pd.DataFrame.from_dict(result_dict).transpose()
    report_name = "{}/limit_of_agreement.csv".format(statistics_results_folder)
    result_dict_df.to_csv(report_name, sep=',', encoding='utf-8')

    return result_dict


def compute_ICC(df1, df2, statistics_results_folder):
    """
    Compute the two-way mixed ICC.

    References
    ----------
    .. [1] R library used for the ICC implementation: http://www.personality-project.org/r/html/ICC.html

    .. [2] R to python: https://rpy2.github.io/doc/latest/html/index.html#

    Notes
    -----
    More info on what is the two-way mixed ICC: https://www.uvm.edu/~dhowell/methods8/Supplements/icc/More%20on%20ICCs.pdf
    """

    psych = importr("psych")
    result_dict = {}
    # Loop over each feature
    for column in df1.columns:
        x = df1[column]
        y = df2[column]

        try:
            r_df = DataFrame({"WBB feature": FloatVector(x),
                              "FP feature": FloatVector(y)})
            # Compute the two way mixed ICC
            icc_res = psych.ICC(r_df)
            iccs_r_df = icc_res[0]
            iccs_df = pandas2ri.ri2py(iccs_r_df)

            # Select the ICC that corresponds to the 2 way mixed model (see links above)
            icc = iccs_df.iloc[5]["ICC"]
            icc_lower_bound = iccs_df.iloc[5]["lower bound"]
            icc_upper_bound = iccs_df.iloc[5]["upper bound"]
            icc_result = "{}({}, {})".format(round(icc, 4), round(icc_lower_bound, 4), round(icc_upper_bound, 4))

            # Store the results
            result_dict[column] = {}
            result_dict[column]["ICC"] = icc_result

        except (RuntimeWarning, Exception) as err:
            logger.error("Problem with feature: {}.\n{}".format(column, err), exc_info=True, stack_info=True)
            pass

    # Save the results
    result_dict_df = pd.DataFrame.from_dict(result_dict).transpose()
    report_name = "{}/ICC.csv".format(statistics_results_folder)
    result_dict_df.to_csv(report_name, sep=',', encoding='utf-8')

    return result_dict
