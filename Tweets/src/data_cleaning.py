
import pandas as pd


class DataCleaning:


	@staticmethod
	def load_data(): 
		positive = pd.DataFrame()
		negative = pd.DataFrame()
		neutral = pd.DataFrame()

		positive["positive"] = pd.read_csv('../data/raw/processedPositive.csv', header=None ).T.squeeze()
		negative["negative"] = pd.read_csv('../data/raw/processedNegative.csv', header=None).T.squeeze()
		neutral["neutral"] = pd.read_csv('../data/raw/processedNeutral.csv', header=None).T.squeeze()

		merged = pd.merge(positive, negative, left_index=True, right_index=True)
		data_frame = pd.merge(merged, neutral, left_index=True, right_index=True)


		data_frame = data_frame.fillna('').astype(str)
		return data_frame