import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def encode_col(df, col, fillna = 'NA'):
    #Encodes a categorical variable as integers
    if fillna != None: df[col].fillna(fillna)
    col_vals = pd.unique(df[col])
    if len(col_vals) == 2:
        df[col] = df[col] == df[col].iloc[0]
        return
    for col_val, ival in zip(col_vals, np.arange(1, len(col_vals) + 1)):
        df[col][df[col] == col_val] = ival

class TranslationTest():
    def __init__(self, datadir = './data', loaddata = True, mergedata = True):
        if loaddata:
            self.user = pd.read_csv('{}/user_data_table.csv'.format(datadir))
            self.df = pd.read_csv('{}/test_data_table.csv'.format(datadir))
        if mergedata:
            self.df = pd.merge(self.df, self.user, on = ['user_id'])

    def compare_value_counts(self,sub_vals, sub_col, valcnt_col):
        for sub_val in sub_vals:
            print '{} = {}:'.format(sub_col, sub_val)
            tmp_df = self.df[self.df[sub_col] == sub_val]
            print tmp_df[valcnt_col].value_counts()*1./tmp_df.shape[0]

    def naive_test(self, df = None, excludeFrance = True, iterate_on = None, return_err = True):
        if np.shape(df) == (): df = self.df
        if excludeFrance: df = df[df.country != 'France']
        if iterate_on != None: 
            iterate_vals = pd.unique(df[iterate_on])
            for iterate_val in iterate_vals:
                print "\n{} = {}:".format(iterate_on, iterate_val)
                self.naive_test(df = df[df[iterate_on] == iterate_val], excludeFrance = excludeFrance, iterate_on = None, return_err = return_err)
            print '\nFull Test:'
        test, control = df[df.test == 1], df[df.test == 0]
        try:
            test_conv_rate = test.conversion.sum()*1./test.shape[0]
        except:
            test_conv_rate = 0
        control_conv_rate = control.conversion.sum()*1./control.shape[0]
        if return_err:
            test_conv_err, control_conv_err = np.sqrt(test_conv_rate * (1 - test_conv_rate)/ test.shape[0]), np.sqrt(control_conv_rate * (1 - control_conv_rate)/ control.shape[0])
            test_err_str, control_err_str = ' +/- {:.2f}%'.format(test_conv_err * 100), ' +/- {:.2f}%'.format(control_conv_err * 100)
        else:
            test_err_str, control_err_str = '', ''
        print '   Test Conversion Rate: {:.2f}%{}'.format(test_conv_rate * 100, test_err_str)
        print 'Control Conversion Rate: {:.2f}%{}'.format(control_conv_rate * 100, control_err_str)

    def FeatureCorrelationsPlot(self, df = None, cols = ['conversion', 'date', 'source', 'device', 'browser_language', 'ads_channel', 'browser', 'sex', 'age', 'country'], encode_cols = ['source', 'device', 'browser_language', 'ads_channel', 'browser', 'sex', 'country']):
        if np.shape(df) == (): df = self.df.copy()
        df = df[cols]
        if 'date' in cols:
            try:
                df.date = (pd.to_datetime(df.date) - pd.to_datetime(df.date).iloc[0]).dt.total_seconds()/86400
            except:
                pass
        for col in encode_cols: encode_col(df, col)
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14,12))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
        plt.show(1, block = False)

    def find_outliers(self, subdivide_col = 'country', excludeFrance = True, outlier_thresh = 3):
        df = self.df.copy()
        if excludeFrance: df = df[df.country != 'France']
        ConversionRates = self.df.groupby('country')['test'].sum()*1./self.df.groupby('country')['test'].count()
        med, mad = ConversionRates.median(), ConversionRates.mad()
        is_outlier = np.abs(ConversionRates.values - med) / mad > outlier_thresh
        return pd.DataFrame({'Conversion Rate': ConversionRates, 'Outlier': is_outlier})

    def is_there_an_outlier(self, subdivide_col = 'country', excludeFrance = True, outlier_thresh = 3):
        df_outliers = find_outliers(subdivide_col = subdivide_col, excludeFrance = excludeFrance, outlier_thresh = outlier_thresh)
        return np.count_nonzero(df_outliers.Outlier.values) > 0
