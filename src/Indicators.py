import os
import pandas as pd
import math
import numpy as np

period = 6

def stochasticK(df, ndays): 
	low = df['Low'].rolling(window=ndays*period).min()
	high = df['High'].rolling(window=ndays*period).max()
	df['%STOK'] = 100 * (df['Close'] - low)/(high - low)

def momentum(df):
	df['Momentum'] = df['Close'].diff(4)

def ROC(df, ndays):
	df['ROC'] = 100 * (df['Close']/df['Close'].shift(ndays))

def LWR(df, ndays):
	low = df['Low'].rolling(window=ndays*period).min()
	high = df['High'].rolling(window=ndays*period).max()
	df['LW%R'] = 100 * (high - df['Close'])/(high - low)

def aoOscillator(df, ndays):
	df['A/O'] = (df['High'] - df['Close'].shift())/(df['High'] - df['Low'])

def disparity5(df):
	ma5 = df['Close'].rolling(window=5*period).mean()
	df['Disparity5'] = 100 * (df['Close']/ma5)

def disparity10(df):
	ma10 = df['Close'].rolling(window=10*period).mean()
	df['Disparity10'] = 100 * (df['Close']/ma10)

def OSCP(df):
	ma5 = df['Close'].rolling(window=5*period).mean()
	ma10 = df['Close'].rolling(window=10*period).mean()
	df['OSCP'] = (ma5 - ma10)/ma5

def CCI(df, ndays):
	df['Mt'] = (df['Close'] + df['High'] + df['Low'])/3
	df['SMt'] = df['Mt'].rolling(window=ndays*period).mean()
	#df['Dt'] = 
	pass

def RSI(df):
	pass

def OBV(df):
	#df['Closeprev'] =df['Close'].shift(-1)
	df['theta'] = df['Close'] < df['Close'].shift()
	df['theta'].replace([True, False], [1, -1], inplace=True)
	print df['theta']
	df['OBV'] = df['theta'] * df['Volume']
	df['OBV'] = df['OBV'] + df['OBV'].shift()
	del df['theta']

def MA5(df):
	df['MA5'] = df['Close'].rolling(window=5*period).mean()

def BIAS6(df):
	df['BIAS6_%'] = (df['Close'] - df['Close'].rolling(window=6*period).mean())/(df['Close'].rolling(window=6*period).mean())

def PSY12(df):
	pass

def SY(df):
	df['SY'] = 100 * (np.log(df['Close']) - np.log(df['Close'].shift()))

def ASY5(df):
	df['ASY5'] = df['SY'].rolling(window=5*period).mean()

def ASY4(df):
	df['ASY4'] = df['SY'].rolling(window=4*period).mean()

def ASY3(df):
	df['ASY3'] = df['SY'].rolling(window=3*period).mean()

def ASY2(df):
	df['ASY2'] = df['SY'].rolling(window=2*period).mean()

def ASY1(df):
	df['ASY5'] = df['SY'].shift()

if __name__ == '__main__':

	fileDir = os.path.dirname(os.path.realpath('__file__'))
	filename = os.path.join(fileDir, 'Indicators/UNH.csv')
	df = pd.read_csv('UNH.csv')

	#print df.ix[2]['Close']

	stochasticK(df, 5)
	momentum(df)
	ROC(df, 5)
	LWR(df, 5)
	aoOscillator(df, 5)
	disparity5(df)
	disparity10(df)
	OSCP(df)
	OBV(df)
	MA5(df)
	BIAS6(df)
	SY(df)
	ASY5(df)
	ASY4(df)
	ASY3(df)
	ASY2(df)
	ASY1(df)

	print df
	df.to_csv(filename)
	#print df[['Close', 'MA5']]

