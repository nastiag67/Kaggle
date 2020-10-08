import pandas as pd
from pandas_profiling import ProfileReport

from utils import loader


class LoadDataset:
    """Loads data of different formats:
        - one .xlsx, .csv (.dsv),
        - several .xlsx, .csv (.dsv),
        - using a simple sql query,
        - using an sql query which loads data based on a list of IDs

    Format of the data can be specified in the following optional parameters:
        - multiple_csv,
        - multiple_xlsx,
        - sql_simle,
        - sql_chunks

    Parameters
    ----------
    filename : string
        Name of the input file

    multiple_csv : bool, default=False
        True if the dataset is in several .csv or .dsv files

    dates : str, default=None
        Provided that multiple_csv = True, specifies a column name with dates to parse

    columns : str, default=None
        Provided that multiple_csv = True, specifies which columns to use

    path : str, default=None
        Provided that multiple_csv = True or multiple_xlsx = True, specifies path to folder with data files

    datatypes : dict, default=None
        Provided that multiple_csv = True, specifies types of the data for faster loading

    multiple_xlsx : bool, default=False
        True if the dataset is in several .xlsx files

    sql_simle : bool, default=False
        True if the dataset is loaded from a database

    query : str, default=None
        Provided that sql_simle=True or sql_chunks=True, specifies a query of the data to load

    driver : str, default=None
        Provided that sql_simle=True or sql_chunks=True, specifies a name of the driver

    user : str, default=None
        Provided that sql_simle=True or sql_chunks=True, specifies a username

    pw : str, default=None
        Provided that sql_simle=True or sql_chunks=True, specifies a password

    sql_chunks : bool, default=False
        True if the dataset is loaded from a database in several  sublists of IDs

    lst : str, default=None
        Provided that sql_chunks = True, specifies a list of IDs

    size : str, default=None
        Provided that sql_chunks = True, specifies size of a chunk.

    """
    def __init__(self, filename,
                 multiple_csv=False, dates=None, columns=None, path=None, datatypes=None,
                 multiple_xlsx=False,
                 sql_simle=False, query=None, driver=None, user=None, pw=None,
                 sql_chunks=False, lst=None, size=1000):
        self.filename = filename

        if filename.find('.csv') >= 0 and multiple_csv==False:
            self.df = pd.read_csv(filename)
        elif filename.find('.xlsx') >= 0 and multiple_xlsx==False:
            self.df = pd.read_excel(filename)
        elif multiple_csv:
            assert dates is not None and columns is not None and path is not None and datatypes is not None, \
                "The following arguments must be specified to load csv files: dates, columns, path, datatypes"
            self.df = loader.csv_loader(dates, columns, path, datatypes, sep='|', dec=',', encoding='utf-8')
        elif multiple_xlsx:
            assert path is not None, \
                "The following arguments must be specified to load xlsx files: path"
            self.df = loader.excel_loader(path, dates)
        elif sql_simle:
            assert query is not None and driver is not None and user is not None and pw is not None, \
                "The following arguments must be specified to load sql files: query, driver, user, pw"
            self.df = loader.sql_loader(query, driver, user, pw)
        elif sql_chunks:
            assert query is not None and driver is not None and user is not None and pw is not None, \
                "The following arguments must be specified to load sql files: query, driver, user, pw, lst"
            self.df = loader.chunk_loader(query, driver, user, pw, lst, size)
        else:
            raise AttributeError('Unknown file format to load')

    def get_data(self):
        """ Returns a loaded dataframe"""
        return self.df

    def get_randomdata(self, n=None, frac=None):
        """ Returns n or a fraction of randomly chosen rows.

        Parameters:
        ----------
        n : int, optional, default=None
            Number of items from axis to return. Cannot be used with `frac`.

        frac : float, optional, default=None
            Fraction of axis items to return. Cannot be used with `n`.

        Returns
        ----------
        df_sample : dataframe

        """
        if n is not None or frac is not None:
            # Randomly sample num_samples elements from dataframe
            df_sample = self.df.sample(n=n, frac=frac).iloc[:, 1:]
        else:
            df_sample = self.df.sample(n=100).iloc[:, 1:]
        return df_sample

