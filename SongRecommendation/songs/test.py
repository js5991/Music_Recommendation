from . import featureSelection
import unittest
import os.path
import pandas as pd


class Test(unittest.TestCase):
    """
    This is the class for unit tests.
    """

    def test_featureselection_init(self):
        """
        Unit test for the constructor of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        temp_test_data_df1 = temp_test_data.ix[:, :"tags"]
        temp_test_data_df2 = temp_test_data.ix[:, "Bay Area":]
        test_data = featureSelection.FeatureSelection(temp_test_data)
        self.assertEqual(test_data.df1.shape[0], temp_test_data_df1.shape[0])
        self.assertEqual(test_data.df2.shape[0], temp_test_data_df2.shape[0])
        self.assertEqual(test_data.df1.shape[1], temp_test_data_df1.shape[1])
        self.assertEqual(test_data.df2.shape[1], temp_test_data_df2.shape[1])

    def test_remove_tag_less_than_threshold(self):
        """
        Unit test for the remove_tag_less_than_threshold function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.remove_tag_less_than_threshold(threshold=0)
        self.assertEqual(test_data.df2.shape, (11, 40))

    def test_return_data_set(self):
        """
        Unit test for the return_data_set function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.return_data_set()
        self.assertEqual(test_data.df1.shape[0], 11)
        self.assertEqual(test_data.df2.shape[0], 11)
        self.assertEqual(test_data.df1.shape[1] + test_data.df2.shape[1], 100)

    def test_to_csv_file(self):
        """
        Unit test for the to_csv_file function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.to_csv_file('test_feature_selected_data.csv')
        self.assertTrue(os.path.isfile('test_feature_selected_data.csv'))






