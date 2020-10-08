import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from pandas_profiling import ProfileReport

from utils import tools as t


class Preprocess:
    """Exploratory Data Analysis of the input dataframe.

    Parameters
    ----------
    df : dataframe
        Input dataframe which will be analysed further.

    """
    def __init__(self, df):
        self.df = df

    def get_overview(self, n=None, max_rows=1000):
        """ Returns Pandas Profiling report.

        Parameters:
        ----------
        n : int, default=None
            Number of items from axis to return.

        max_rows : int, default=1000
            Number rows on which the ProfileReport is based.

        Notes
        ----------
        Due to technical limitations, the optimal maximum number of rows on which the report is based is 1000.
        If the actual number of rows is higher than 1000, then the report is constructed on randomly chosen 1000 rows.

        Returns
        ----------
        ProfileReport in html.

        """
        # max_rows = 1000  # the optimal maximum number of rows on which the report is based
        if n is None and self.df.shape[0] <= max_rows:
            return ProfileReport(self.df, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width': True}})
        elif n is None and self.df.shape[0] > max_rows:
            print(f"Data is too large (> {max_rows} rows), getting overview for {max_rows} random samples")
            data = self.get_randomData(n=max_rows)
            return ProfileReport(data, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width':True}})
        else:
            data = self.get_randomData(n=n)
            return ProfileReport(data, title='Pandas Profiling Report', minimal=True, html={'style':{'full_width':True}})

    def get_summary(self,
                    nan=True,
                    formats=True,
                    categorical=True,
                    min_less_0=True,
                    check_normdist=True):
        """Describes the data.

        Parameters:
        ----------
        nan : bool, default=True
            True if need to return a list of NaNs.

        formats : bool, default=True
            True if need to return all the formats of the columns.

        categorical : bool, default=True
            True if need to return values which can be categorical.
            Variable is considered to be categorical if there are less uique values than num_ifcategorical.

        min_less_0 : bool, default=True
            True if need check for variables which have negative values.

        check_normdist : bool, default=True
            True if need check actual distribution against Normal distribution.
            Will make plots of each variable considered against the Normal distribution.

        Returns
        ----------
        A description of the data in text format and plots if check_normdist=True.

        """
        # Checking for NaN
        if nan:
            nans = list(
                pd.DataFrame(self.df.isna().sum()).rename(columns={0: 'NaNs'}).reset_index().query("NaNs>0")['index'])
            t.log(t.black('NaNs: '), nans)
        else:
            nans = False

        # Checking for unique formats
        if formats:
            unique_formats = list(self.df.dtypes.unique())
            t.log(t.black('Unique formats: '), unique_formats)
        else:
            formats is False

        # Checking for possible categorical values
        if categorical:
            num_ifcategorical = 10
            possibly_categorical = []
            for col in self.df.columns:
                set_unique = set(self.df[col])
                if len(set_unique) <= num_ifcategorical:
                    possibly_categorical.append(col)
            t.log(t.black(f'Possibly categorical variables (<{num_ifcategorical} unique values): '), possibly_categorical)
        else:
            categorical is False

        # Checking if min value is < 0
        if min_less_0:
            lst_less0 = list(pd.DataFrame(self.df[self.df < 0].any()).rename(columns={0: 'flag'}).query("flag==True").index)
            t.log(t.black(f'Min value < 0: '), lst_less0)
        else:
            min_less_0 is False

        # Plotting actual distributions vs Normal distribution
        def check_distribution(columns, plot_cols=6):
            plt.style.use('seaborn-white')

            if plot_cols > len(columns) - 2:
                t.log(t.yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns.")
                plot_cols = len(columns) - 2

            # figure size = (width,height)
            f1 = plt.figure(figsize=(30, len(columns) * 3))

            total_plots = len(columns)
            rows = total_plots - plot_cols

            for idx, y in enumerate(columns):
                if len(set(self.df[y])) >= 3:
                    idx += 1
                    ax1 = f1.add_subplot(rows, plot_cols, idx)
                    ax1.set_xlabel(y)
                    sns.distplot(self.df[y],
                                 color='b',
                                 hist=False
                                 )
                    # parameters for normal distribution
                    x_min = self.df[y].min()
                    x_max = self.df[y].max()
                    mean = self.df[y].mean()
                    std = self.df[y].std()
                    # plotting normal distribution
                    x = np.linspace(x_min, x_max, self.df.shape[0])
                    y = scipy.stats.norm.pdf(x, mean, std)
                    plt.plot(x, y, color='black', linestyle='dashed')

        if check_normdist:
            t.log(t.black('Plotting distributions of variables against normal distribution:'))
            check_distribution(self.df.columns, plot_cols=6)

        return nans, formats, categorical, min_less_0


class Outliers(Preprocess):
    """Performs outliers detection and analysis.

    Parameters
    ----------
    df : dataframe
        Input dataframe which will be analysed further.

    """
    def __init__(self, df):
        Preprocess.__init__(self, df)

    def _z_score(self, columns, threshold=3):
        """Detects outliers based on z-score.

        Parameters:
        ----------
        columns : str
            A string of columns which will be analysed together using z-score.

        threshold : int, default=3
            Threshold against which the outliers are detected.

        Returns
        ----------
        df_outliers_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        """
        # remove outliers based on chosen columns
        df_selected = self.df[columns].copy()

        # remove outliers
        z = np.abs(stats.zscore(df_selected))

        df_outliers_clean = self.df[(z < threshold).all(axis=1)]

        # get outliers df
        df_outliers = self.df[~self.df.index.isin(df_outliers_clean.index)]

        return df_outliers_clean, df_outliers

    def _IQR(self, columns, q1=0.25):
        """Detects outliers based on interquartile range (IQR).

        Parameters:
        ----------
        columns : str
            A string of columns which will be analysed together using IQR.

        q1 : float, default=0.25
            Threshold against which the outliers are detected.

        Returns
        ----------
        df_outliers_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        """
        # remove outliers based on chosen columns
        df_selected = self.df[columns]

        # remove outliers
        Q1 = df_selected.quantile(q1)
        Q3 = df_selected.quantile(1 - q1)
        IQR = Q3 - Q1

        df_outliers_clean = self.df[~((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR))).any(axis=1)]

        # get outliers df
        df_outliers = self.df[~self.df.index.isin(df_outliers_clean.index)]

        return df_outliers_clean, df_outliers

    def _plot(self, columns, df_clean, df_outliers, plot_cols=6):
        """Plots the dataframe and marks the outliers by a red cross.

        Parameters:
        ----------
        columns : str
            A string of columns which will be plotted.

        df_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        plot_cols : int, default=6
            Determines how many columns the plots will form.

        """
        plt.style.use('seaborn-white')

        if plot_cols > len(columns) - 2:
            t.log(t.yellow('ERROR: '), f"Can't use more than {len(columns) - 2} columns in one row.")
            plot_cols = len(columns) - 2

        # figure size = (width,height)
        f1 = plt.figure(figsize=(30, len(columns) * 3))

        total_plots = len(columns)
        rows = total_plots - plot_cols

        for idx, y in enumerate(columns):
            idx += 1
            ax1 = f1.add_subplot(rows, plot_cols, idx)
            sns.regplot(x=df_clean.index,
                        y=y,
                        data=df_clean,
                        scatter=True,
                        fit_reg=False,
                        color='lightblue',
                        )
            sns.regplot(x=df_outliers.index,
                        y=y,
                        data=df_outliers,
                        scatter=True,
                        fit_reg=False,
                        marker='x',
                        color='red',
                        )

    def show(self, columns, how='z_score', show_plot=False, **kwargs):
        """Detects outliers using one of the available methods.

        Parameters:
        ----------
        columns : str
            A string of columns which will be analysed together.

        how : str, default=z_score
            Method using which the outliers are detected.

        show_plot : bool, default=False
            True if need to see the plot of the data with the marked outliers.

        **kwargs
            Specifies extra arguments which may be necessary for one of the methods of finding outliers:

            threshold : int, default=3
                True if need to return all the formats of the columns.

            q1 : float, default=0.25
                True if need to return all the formats of the columns.

        Returns
        ----------
        df_clean : dataframe
            Dataframe without outliers.

        df_outliers : dataframe
            Dataframe of outliers.

        df : dataframe
            Original dataframe with outliers.
            Contains a new column called 'outliers' (bool) where the outliers are flagged (True if outlier).

        """
        if how == 'z_score':
            assert 'threshold' in kwargs, 'To use z-score method, threshold must be specified (default = 3)'
            df_clean, df_outliers = self._z_score(columns, kwargs['threshold'])
        elif how == 'IQR':
            assert 'q1' in kwargs, 'To use z-score method, q1 must be specified (default = 0.25)'
            df_clean, df_outliers = self._IQR(columns, kwargs['q1'])
        else:
            raise AttributeError('Unknown outlier detection method. Existing methods: z_score, IQR')

        df = self.df.copy()
        df['outliers'] = df.index.isin(df_outliers.index).copy()

        if show_plot:
            self._plot(columns, df_clean, df_outliers)

        return df_clean, df_outliers, df

