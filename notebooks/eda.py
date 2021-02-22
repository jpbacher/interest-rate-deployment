import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def quick_observation(df, target, duplicate_col=None):

    """A quick look at certain characteristics of a dataset.

    Argument:
    df: the dataframe
    subset: a string, the columns to look for duplicate entries

    Returns:
    Prints various properties of the dataframe.
    """

    print(f'------The shape of the uncleaned training data: {df.shape}\n')
    print(f'------The datatypes of the predictors and target:\n{df.dtypes}\n')
    print(f'------The number of duplicates: {df.duplicated(subset=duplicate_col).sum()}\n')
    print(f'------Class imbalance: \n{df[target].value_counts() / df.shape[0]}\n')
    print(f'------Missing features:\n{df.isnull().sum()}')

def plot_target(df, tgt):
    '''Creates a simple bar chart of the response variable's value counts'''
    plt.figure(figsize=(12, 6))
    sns.barplot(df[tgt].value_counts(), df[tgt].dropna().unique(), orient='h')
    sns.despine(left=True)
    plt.xlabel('Count')
    plt.title('Distribution of %s' % str.capitalize(tgt))
    plt.show()
    print('Percentage of each class:\n{}'.format(df[tgt].value_counts() / df.shape[0]))

def get_missing_features(df, thresh):
    nulls = df.isnull().sum()
    n = nulls[nulls > 0]
    missing = pd.DataFrame(n, columns=['total'])
    missing['percentage'] = missing['total'] / df.shape[0]
    missing.sort_values('percentage', ascending=False, inplace=True)
    columns = missing.loc[missing['percentage'] >= thresh].index
    print(len(columns))
    return columns, missing

def get_no_variance_features(df):
    """Look for features that do not contain any variance and thus, do not
    provide any predictive power.
    Parameters:
    df: the dataframe
    Returns:
    list of no variance features & prints the size of the list
    """
    no_variance_features = []
    for column in df.columns:
        if df[column].unique().size < 2 or df[column].unique().size == 2 and np.nan in list(df[column].unique()):
            no_variance_features.append(column)
    print(f'Number of features that do not provide any variance: {len(no_variance_features)}')
    if len(no_variance_features) > 0:
        print(f'{no_variance_features}')
        return no_variance_features    

def get_high_correlated_features(df, correlated_thresh):
    """Get high correlated features from the dataset.
    Parameters
    ----------
    df: the dataframe
    correlated_thresh: float, between 0 & 1 - only return features that are greater than this threshold
    Returns
    -------
    A list of correlated features
    """
    corr_matrix = df.corr().abs()
    upper_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    correlated_features = [feat for feat in upper_matrix.columns if any(upper_matrix[feat] > correlated_thresh)]
    if len(correlated_features) > 0:
        print(f'Correlated feature above {correlated_thresh}:\n{correlated_features}')
        return correlated_features


class Visualizations:
    def __init__(self, df, target):
        """
        Arguments:
        df: pandas DataFrame
        target: a string, the response variable from the df
        """
        assert isinstance(df, pd.DataFrame), 'df must be a pandas DataFrame'
        assert isinstance(target, str), 'target must be a string from the df specified'
        self.df = df
        self.target = target
        self.color = 'darkred'
        self.palette = 'OrRd'
        self._get_target_plot(self.df, self.target)

    def _get_target_plot(self, df, target):
        '''Creates a simple bar chart of the response variable's value counts'''
        plt.figure(figsize=(12, 6))
        sns.barplot(df[target].value_counts(), 
                    df[target].dropna().unique(), 
                    palette=self.palette,
                    orient='h')
        sns.despine(left=True)
        plt.xlabel('Count')
        plt.title('Distribution of %s' % str.capitalize(target))
        plt.savefig('class_imbalance.png')
        plt.show()
        print('Percentage of each class:\n{}'.format(df[target].value_counts() / df.shape[0]))


    @staticmethod
    def get_numerical_features(df, target):
        """Get a list of the numerical features
        Arguments:
        df: the dataframe
        target: the response variable from the dataframe
        Returns:
        a list of the numerical features from the dataframe
        """
        numerical_features = df.dtypes[(df.dtypes == np.float) | (df.dtypes == np.integer)].index
        numerical_features = [feature for feature in numerical_features if target not in feature]
        assert len(numerical_features) > 0, 'There are are no numerical features'
        return numerical_features

    @staticmethod
    def get_categorical_features(df, target):
        """Get a list of the categorical features
        Arguments:
        df: the dataframe
        target: the response variable from the dataframe
        Returns:
        a list of the categorical features from the dataframe
        """
        cat_features = df.dtypes[(df.dtypes == 'object') | (df.dtypes == 'category')].index
        cat_features = [feature for feature in cat_features if target not in feature]
        assert len(cat_features) > 0, 'There are no categorical features'
        return cat_features

    def _get_cat_mean(self, df, feature, target):
        return (df.groupby(feature)[target].mean()).sort_values(ascending=False)

    def plot_features(self, df, target, data_fraction=1.0, max_unique_values=100):
        """Plot distributions of a feature; for continuous variables, plot data of the response variable
        to that feature; if categorical, plot a boxplot of target variable to the nominal
        variables' unique values
        Arguments:
        df: the dataframe
        target: the response variable
        data_fraction: the fraction of samples to use when plotting the data for continuous features (to
        decrease time of plotting every point)
        max_unique_values: if a categorical variables' unique values above a threshold, will not plot
        that feature
        Returns:
        The plots of all features in the dataframe
        """
        majority_df = df.loc[df[target] == 0]
        minority_df = df.loc[df[target] == 1]
        label = ['Died', 'Survived']
        print('---Numerical Features---')
        for feat in self.get_numerical_features(df, target):
            fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
            box = sns.boxplot(x=df[feat].dropna(), y=df[target], orient='h', showmeans=True, ax=axs[0])
            sns.kdeplot(majority_df[feat].dropna(), 
                        shade=True, 
                        label=label[0],
                        ax=axs[1])
            kde = sns.kdeplot(minority_df[feat].dropna(), 
                              shade=True, 
                              label=label[1],
                              ax=axs[1])
            kde.legend(labels=label, loc=0)
            sns.despine(left=True)
            kde.set(xlabel=feat, ylabel='', yticks=([]), title='%s Distribution' % str.capitalize(feat))
            box.set(xlabel=feat, ylabel=target, title='%s Comparison' % str.capitalize(feat))
            #plt.savefig(f'{feat}.png')
            plt.show()
        print('---Categorical Features---')
        for feat in self.get_categorical_features(df, target):
            if len(df[feat].unique()) <= max_unique_values:
                plt.figure(figsize=(7, 8))
                sns.barplot(x=(round(self._get_cat_mean(df, feat, target), 4) * 100).values,
                            y=self._get_cat_mean(df, feat, target).index, orient='h')
                sns.despine(left=True)
                plt.xlabel('Percentage (%)')
                plt.ylabel('')
                plt.title('Percentage of %s with respect to %s' %
                          (str.capitalize(target), str.capitalize(feat)))
                #plt.savefig(f'{feat}.png')
                plt.show()

    def plot_new_feature(self, df, target, feat):
        if df[feat].dtype == 'O':
            plt.figure(figsize=(7, 8))
            sns.barplot(x=(round(self._get_cat_mean(df, feat, target), 4) * 100).values,
                           y=self._get_cat_mean(df, feat, target).index, orient='h')
            sns.despine(left=True)
            plt.xlabel('Percentage (%)')
            plt.ylabel('')
            plt.title('Percentage of %s with respect to %s' %
                     (str.capitalize(target), str.capitalize(feat)))
            plt.savefig(f'{feat}.png')
            plt.show()
    
    @staticmethod
    def plot_heatmap(df, cmap='OrRd', annot=False):
        """Plot a correlation matrix of the data.
        Arguments:
        df: the dataframe
        cmamp: color
        annot: boolean, whether to show correlation value
        """
        plt.figure(figsize=(15, 15))
        sns.heatmap(df.corr(), cmap=cmap, annot=annot)
        plt.title('Correlation Matrix')
        plt.show()