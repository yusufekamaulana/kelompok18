from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import datetime
import math
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from plotly import graph_objs as go
import numpy as np
import pandas_ta as ta

app = Flask(__name__, template_folder="templates", static_folder="static")

def load_data(ticker,START,TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def preprocessData(aplData):

	def ewm(data, column, window, span):
		result = [0]
		for i in pd.DataFrame.rolling(data[column], window):
			result.append(np.mean([k for k in i.ewm(span=span, adjust=False).mean()]))
		return result[:-1]

	def autocorr(array, column, window, lag=2):
		w = window + lag
		result = [0] * (w)
		print(array.shape[0])
		for i in range(w, array.shape[0]):
			data = array[column][i - w:i]
			d = data
			y_bar = np.mean(data)
			denominator = sum([(i - y_bar) ** 2 for i in data])
			lagData = [i - y_bar for i in d][lag:]
			actualData = [i - y_bar for i in d][:-lag]
			numerator = sum(np.array(lagData) * np.array(actualData))
			result.append((numerator / denominator))

		return result

	def doubleExSmoothing(array, column, window, trend):
		result = [0] * (window)
		for i in range(window, array.shape[0]):
			data = array[column][i - window:i]
			values = ExponentialSmoothing(data, trend=trend).fit().fittedvalues
			d = [i for i in values.tail(1)]
			result.append(d[0])

		return result


	aplData['Close'] = aplData['Close'].shift(-1)
	moving_AverageValues = [10, 20, 50]
	for i in moving_AverageValues:
		column_name = "MA_%s" % (str(i))
		aplData[column_name] = pd.DataFrame.rolling(aplData['Close'], i).mean().shift(1)
	aplData['5_day_std'] = aplData['Close'].rolling(window=5).std().shift(1)
	aplData['Daily Return'] = aplData['Close'].pct_change().shift(1)
	aplData['SD20'] = aplData.Close.rolling(window=20).std().shift(1)
	aplData['Upper_Band'] = aplData.Close.rolling(window=20).mean().shift(1) + (aplData['SD20'] * 2)
	aplData['Lower_Band'] = aplData.Close.rolling(window=20).mean().shift(1) - (aplData['SD20'] * 2)
	aplData['Close(t-1)'] = aplData.Close.shift(periods=1)
	aplData['Close(t-2)'] = aplData.Close.shift(periods=2)
	aplData['Close(t-5)'] = aplData.Close.shift(periods=5)
	aplData['Close(t-10)'] = aplData.Close.shift(periods=10)
	aplData['EMA_10'] = ewm(aplData, "Close", 50, 10)
	aplData['EMA_20'] = ewm(aplData, "Close", 50, 20)
	aplData['EMA_50'] = ewm(aplData, "Close", 50, 50)
	aplData['MACD'] = aplData['EMA_10'] - aplData['EMA_20']
	aplData['MACD_EMA'] = ewm(aplData, "MACD", 50, 9)
	aplData['ROC'] = ((aplData['Close'].shift(1) - aplData['Close'].shift(10)) / (aplData['Close'].shift(10))) * 100
	funct = lambda x: pd.Series(extract_date_features(x))
	aplData[['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
			 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', 'Year', 'Month', "Is_Monday",
			 "Is_Tuesday", "Is_Wednesday", "Is_Thursday", "Is_Friday"]] = aplData["Date"].apply(funct)
	aplData['AutoCorr_1'] = autocorr(aplData, 'Close', 10, 1)
	aplData['AutoCorr_2'] = autocorr(aplData, 'Close', 10, 2)
	aplData['HWES2_ADD'] = doubleExSmoothing(aplData, 'Close', 50, 'additive')

	aplData = aplData.iloc[:-1]
	# aplData = aplData.tail(50)
	aplData.reset_index(inplace=True)
	aplData = aplData.drop(['index'], axis=1)
	return aplData

def extract_date_features(date_val):
	Day = date_val.day
	DayofWeek = date_val.dayofweek
	Dayofyear = date_val.dayofyear
	Week = date_val.week
	Is_month_end = date_val.is_month_end.real
	Is_month_start = date_val.is_month_start.real
	Is_quarter_end = date_val.is_quarter_end.real
	Is_quarter_start = date_val.is_quarter_start.real
	Is_year_end = date_val.is_year_end.real
	Is_year_start = date_val.is_year_start.real
	Is_leap_year = date_val.is_leap_year.real
	day = date_val.weekday()
	Is_Monday = 1 if day == 0 else 0
	Is_Tuesday = 1 if day == 1 else 0
	Is_Wednesday = 1 if day == 2 else 0
	Is_Thursday = 1 if day == 3 else 0
	Is_Friday = 1 if day == 4 else 0
	Year = date_val.year
	Month = date_val.month

	return Day, DayofWeek, Dayofyear, Week, Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start, Is_year_end, Is_year_start, Is_leap_year, Year, Month, Is_Monday, Is_Tuesday, Is_Wednesday, Is_Thursday, Is_Friday

def extractFeatures(lastFeatures, y_train, date):
	'''Function used to extract input features based on previous close price and date value
     lastFeatures - last few values (features)
     y_train - close price values
     date - current date
     return - all input features '''



	def autocorr(array, window, lag=2):
		w = window + lag

		data = array
		d = data
		y_bar = np.mean(data)
		denominator = sum([(i - y_bar) ** 2 for i in data])
		lagData = [i - y_bar for i in d][lag:]
		actualData = [i - y_bar for i in d][:-lag]
		numerator = sum(np.array(lagData) * np.array(actualData))

		return numerator / denominator

	def doubleExSmoothing(array, trend):
		data = array
		values = ExponentialSmoothing(data, trend=trend).fit().fittedvalues
		d = [i for i in values.tail(1)]
		#     result.append( d[0])

		return d[0]

	currentData = {i: {None} for i in lastFeatures.columns}
	if lastFeatures.shape[0] > 50 or True:
		lastFeatures = lastFeatures.tail(50)
		lastValue = lastFeatures.iloc[-1]

		end_date = date

		day = pd.to_datetime(end_date).weekday()
		if day <= 4:

			currentData['MA_10'] = y_train.tail(10).mean()
			currentData['MA_20'] = y_train.tail(20).mean()
			currentData['MA_50'] = y_train.tail(50).mean()
			currentData['5_day_std'] = y_train.tail(5).std()
			#             currentData['SD20'] = y_train.tail(20).rolling(window=20).std().iloc[-1]
			currentData['SD20'] = y_train.tail(20).std()
			currentData['Daily Return'] = y_train.tail(2).pct_change().iloc[-1]
			currentData['Upper_Band'] = (y_train.tail(20).mean() + (currentData['SD20'] * 2))
			currentData['Lower_Band'] = (y_train.tail(20).mean() - (currentData['SD20'] * 2))
			#             print(currentData['Upper_Band'])
			currentData['Close(t-1)'] = y_train.iloc[-1]
			currentData['Close(t-2)'] = y_train.iloc[-2]
			currentData['Close(t-5)'] = y_train.iloc[-5]
			currentData['Close(t-10)'] = y_train.iloc[-10]

			#             aplData['EMA_10'] = ewm(aplData, "Close",50,10)
			currentData['EMA_10'] = np.mean(y_train.tail(50).ewm(span=10, adjust=False).mean())
			currentData['EMA_20'] = np.mean(y_train.tail(50).ewm(span=20, adjust=False).mean())


			currentData['EMA_50'] = np.mean(y_train.tail(50).ewm(span=50, adjust=False).mean())
			currentData['MACD'] = currentData['EMA_10'] - currentData['EMA_20']
			currentData['MACD_EMA'] = np.mean(lastFeatures['MACD'].tail(50).ewm(span=9, adjust=False).mean())
			#             currentData['MACD_EMA'] = lastFeatures['MACD'].tail(50).ewm(span=9, adjust=False).mean().iloc[-1]
			currentData['ROC'] = ((y_train.iloc[-1] - y_train.iloc[-10]) / (y_train.iloc[-10])) * 100
			result = list(extract_date_features(end_date))

			for i, v in enumerate(
					['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
					 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', "Year", 'Month', "Is_Monday",
					 "Is_Tuesday", "Is_Wednesday", "Is_Thursday", "Is_Friday"]):
				currentData[v] = result[i]
			currentData["AutoCorr_1"] = autocorr(y_train.tail(11), 10, lag=1)
			currentData["AutoCorr_2"] = autocorr(y_train.tail(12), 10, lag=2)
			currentData['HWES2_ADD'] = doubleExSmoothing(y_train.tail(50), 'additive')
		else:
			pass


	return currentData

def predictFuture(Nday,X_train,y_train,model,sX_val,sY_val):
	lastFeatures, lastPriceValues = pd.DataFrame(X_train).tail(50), y_train.tail(50)
	date = "{}/{}/{}".format(int(lastFeatures.iloc[-1]['Month']), int(lastFeatures.iloc[-1]['Day']),
							 int(lastFeatures.iloc[-1]['Year']))
	startDate = pd.to_datetime(date) + datetime.timedelta(days=1)
	lastDate = date
	predValues = []
	totalDates = []
	i = 1
	fi = []
	while i <= Nday:
		end_date = pd.to_datetime(date) + datetime.timedelta(days=1)
		date = end_date
		day = pd.to_datetime(end_date).weekday()
		if day <= 4:
			i += 1
			lastDate = end_date
			if i == 2:
				startDate = end_date
			currentData = extractFeatures(lastFeatures, lastPriceValues, end_date)

			df = {k: [v] for k, v in currentData.items()}
			df["Date"] = end_date
			totalDates.append(end_date)

			df = pd.DataFrame(df)
	
			df3 = pd.concat([lastFeatures, df], ignore_index=True)

			inputFeature = df3.drop(["Date"],axis = 1).iloc[-1]

			inpu = np.array([i for i in inputFeature])

			inpu = sX_val.transform(inpu.reshape(1, -1))


			inpu = inpu.reshape(1, X_train.shape[1]-1, 1)


			pred = model.predict(inpu, verbose=0)


			pred = sY_val.inverse_transform(pred.reshape(-1, 1))

			predValues.append(pred[0][0])
			lastFeatures = df3.tail(50)
			lastPriceValues = list(lastPriceValues)
			lastPriceValues.append(pred[0][0])
			lastPriceValues = pd.Series(lastPriceValues)
			lastPriceValues = lastPriceValues.tail(50)


	df3["pred"] = lastPriceValues
	predictions = pd.DataFrame(data = [np.array(totalDates),np.array(predValues)]).T
	predictions.columns = ['Date', 'pred']

	return predictions


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/fitur')
def fitur():
    return render_template('fitur.html')

@app.route('/actual', methods=["GET", "POST"])
def make_chart():
    if request.method == "POST":
        # end_date = date.today().strftime("%Y-%m-%d")
        # start_date = '2021-01-01'
        selected_stock = request.form['emitenSelect']
        # emiten = selected_stock.replace('.JK', ' ')
        ticker = yf.Ticker(selected_stock)
        info = ticker.info
        # Mendapatkan beberapa informasi tertentu
        long_name = info.get('longName')
        sector = info.get('sector')
        business_summary = info.get('longBusinessSummary')
        employees = info.get('fullTimeEmployees')

        df = yf.download(str(selected_stock), period="max")
        df = df.reset_index()
        df_1 = df.copy().drop(columns=['Adj Close'])
        df_1['Date'] = pd.to_datetime(df_1['Date'])
        df_1['Date'] = df_1['Date'].astype(str)
        df_1 = df_1.rename(columns={'Date':'x'})
        jsoon = df_1.to_json(orient='values')
        # redirect(url_for("user", usr=user))
        return render_template("actual.html", selected_stock=selected_stock, jsoon=jsoon, info=info, long_name=long_name, sector=sector, business_summary=business_summary, employees=employees)

    return render_template('actual.html')
    
    # return render_template('home dulu ksrng actual.html', jsoon=jsoon, user_input=user_input)
    

@app.route('/prediksi/', methods=["GET", "POST"])
def prediksi():
    if request.method == "POST":
        TODAY = date.today().strftime("%Y-%m-%d")
        START = "2021-01-01"
        selected_stock = request.form.get('emitenSelect')
        # selected_stock = request.form.get('emitenSelect')
        emiten = selected_stock.replace('.JK', ' ')
        data = load_data(selected_stock, START, TODAY)
        df_train = data[['Date', 'Close']]
		
        if selected_stock == "CPIN.JK":
            lstmModel = tf.keras.models.load_model("cpin.h5")
            # updatedModel = Sequential()
            info = "PT Charoen Pokphand Indonesia Tbk"
        elif selected_stock == "GGRM.JK":
            lstmModel = tf.keras.models.load_model("ggrm.h5")
            info = "PT Gudang Garam Tbk"
        elif selected_stock == "INDF.JK":
            lstmModel = tf.keras.models.load_model("indf.h5")
            info = "PT Indofood Sukses Makmur Tbk"
        elif selected_stock == "UNVR.JK":
            lstmModel = tf.keras.models.load_model("unvr.h5")
            info = "PT Unilever Indonesia Tbk"
        elif selected_stock == "BBCA.JK":
            lstmModel = tf.keras.models.load_model("bbca.h5")
            info = "PT Bank Central Asia Tbk"
        elif selected_stock == "BBNI.JK":
            lstmModel = tf.keras.models.load_model("bbni.h5")
            info = "PT Bank Negara Indonesia (Persero) Tbk"
        elif selected_stock == "BBRI.JK":
            lstmModel = tf.keras.models.load_model("bbri.h5")
            info = "PT Bank Rakyat Indonesia (Persero) Tbk"
        elif selected_stock == "BBTN.JK":
            lstmModel = tf.keras.models.load_model("bbtn.h5")
            info = "PT Bank Tabungan Negara (Persero) Tbk"
        elif selected_stock == "BMRI.JK":
            lstmModel = tf.keras.models.load_model("bmri.h5")
            info = "PT Bank Mandiri (Persero) Tbk"
        elif selected_stock == "BRIS.JK":
            lstmModel = tf.keras.models.load_model("bris.h5")
            info = "PT Bank Syariah Indonesia Tbk"
        
        latestData = pd.DataFrame(df_train)
        latestData = latestData.tail(365)

        Ndays = int(request.form.get("predictSelect"))
        hari = f"Prediksi {Ndays} hari kedepan:"
        # hari = f"Ndays"
        # Ndays = 60

        if Ndays > 0:
            inputData = preprocessData(latestData)
            y_test = inputData["Close"].tail(100)
            x_test = inputData.drop(["Close"], axis=1).tail(100)
            modelData = inputData.iloc[51:-100, :]
            xtrain = modelData.drop(["Close"], axis=1)
            ytrain = modelData["Close"]
            sX_train = MinMaxScaler(feature_range=(0, 1))
            sY_train = MinMaxScaler(feature_range=(0, 1))
            X_train = sX_train.fit_transform(np.array(xtrain.drop(["Date"], axis=1))).reshape(xtrain.shape[0],
                                                                                            xtrain.shape[1] - 1, 1)
            Y_train = sY_train.fit_transform(np.array(ytrain).reshape(-1, 1)).reshape(ytrain.shape[0], )
            sX_val = MinMaxScaler(feature_range=(0, 1))
            sY_val = MinMaxScaler(feature_range=(0, 1))
            X_valLSTM = sX_val.fit_transform(np.array(x_test.drop(["Date"], axis=1))).reshape(x_test.shape[0],
                                                                                            x_test.shape[1] - 1, 1)
            y_valLSTM = sY_val.fit_transform(np.array(y_test).reshape(-1, 1)).reshape(y_test.shape[0], )

            predictions = predictFuture(Ndays, x_test, y_test, lstmModel, sX_val, sY_val)

            profit = ((predictions["pred"].iloc[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
            predictions.set_index('Date', inplace=True)

        predict_test = lstmModel.predict(X_valLSTM)
        predict_test = sY_val.inverse_transform(predict_test)
        rmse = np.sqrt(np.mean(predict_test - y_test.values) ** 2)

        df = df_train.copy()
        df.set_index('Date', inplace=True)
        predictions.columns = ['Close']
        aktualdanprediksi = pd.concat([df, predictions])

        df1 = predictions.copy().reset_index()
        df1 = df1.rename(columns={'Date': 'x'})
        df1['x'] = pd.to_datetime(df1['x'])
        df1['x'] = df1['x'].astype(str)
        json_pred = df1.to_json(orient='values')

        df_ori = df_train.copy()
        df_ori = df_ori.rename(columns={'Date': 'x'})
        df_ori['x'] = pd.to_datetime(df_ori['x'])
        df_ori['x'] = df_ori['x'].astype(str)
        json_ori = df_ori.to_json(orient='values')
        rsi = pd.DataFrame(aktualdanprediksi['Close'].copy())
        rsi['RSI'] = rsi.ta.rsi()
        rsiToday = rsi['RSI'].iloc[len(df)-1].round(1)
        
        return render_template('predict.html', json_pred=json_pred, json_ori=json_ori, selected_stock=selected_stock, emiten=emiten, Ndays=Ndays, hari=hari, info=info, rsiToday=rsiToday)
    
    return render_template('predict.html')

@app.route('/fundamental', methods=["GET", "POST"])
def fundamental():
    selected_stock = None
    selected_stock2 = None
    json_dividen = None
    json_dividen2 = None
    json_financial = None
    json_financial2 = None
    json_modal = None
    json_modal2 = None
    json_ekuitas = None
    json_ekuitas2 = None
    long_name = None
    long_name2 = None

    if request.method == "POST":
        # emiten 1
        selected_stock = request.form.get('emiteninput')
        ticker = yf.Ticker(selected_stock)
        info = ticker.info
        long_name = info.get('longName')
		# emiten 2
        selected_stock2 = request.form.get('emiteninput2')
        ticker2 = yf.Ticker(selected_stock2)
        info2 = ticker2.info
        long_name2 = info2.get('longName')

        if request.form.get('analisis') == "dividen":
            # dividen 1
            dividen = ticker.dividends.tail(10)
            dividen = pd.DataFrame(dividen)
            dividen.reset_index(names='Date',inplace=True)
            dividen['Date'] = [i.strftime('%Y-%m') for i in dividen['Date']]
            json_dividen=dividen.to_json(orient='values')
            # dividen 2
            dividen2 = ticker2.dividends.tail(10)
            dividen2 = pd.DataFrame(dividen2)
            dividen2.reset_index(names='Date',inplace=True)
            dividen2['Date'] = [i.strftime('%Y-%m') for i in dividen2['Date']]
            json_dividen2=dividen2.to_json(orient='values')
            
            return render_template('fundamental1.html', json_dividen=json_dividen, json_dividen2=json_dividen2, long_name=long_name, long_name2=long_name2)

        elif request.form.get('analisis') == "financial":
            # finansial 1
            financial = ticker.quarterly_income_stmt
            df_financial = financial.loc[['Net Income','Operating Revenue','Operating Expense']].transpose()
            df_financial.reset_index(names='Date',inplace=True)
            df_financial['Date'] = [i.strftime('%Y-%m') for i in df_financial['Date']]
            json_financial=df_financial.to_json(orient='values')
            # finansial 2
            financial2 = ticker2.quarterly_income_stmt
            df_financial2 = financial2.loc[['Net Income','Operating Revenue','Operating Expense']].transpose()
            df_financial2.reset_index(names='Date',inplace=True)
            df_financial2['Date'] = [i.strftime('%Y-%m') for i in df_financial2['Date']]
            json_financial2=df_financial2.to_json(orient='values')

            return render_template('fundamental1.html', json_financial=json_financial, json_financial2=json_financial2, long_name=long_name, long_name2=long_name2)

        elif request.form.get('analisis') == "modal":
            # modal 1
            balance_sheet = ticker.quarterly_balance_sheet
            modal = balance_sheet.loc[['Total Debt','Common Stock Equity','Total Capitalization']].transpose()
            modal.reset_index(names='Date',inplace=True)
            modal['Date'] = [i.strftime('%Y-%m') for i in modal['Date']]
            json_modal=modal.to_json(orient='values')
            # modal 2
            balance_sheet2 = ticker2.quarterly_balance_sheet
            modal2 = balance_sheet2.loc[['Total Debt','Common Stock Equity','Total Capitalization']].transpose()
            modal2.reset_index(names='Date',inplace=True)
            modal2['Date'] = [i.strftime('%Y-%m') for i in modal2['Date']]
            json_modal2=modal2.to_json(orient='values')

            return render_template('fundamental1.html', json_modal=json_modal, json_modal2=json_modal2, long_name=long_name, long_name2=long_name2)

        elif request.form.get('analisis') == "ekuitas":
            # ekuitas 1
            balance_sheet = ticker.quarterly_balance_sheet
            ekuitas = balance_sheet.loc[['Stockholders Equity','Retained Earnings','Common Stock']].transpose()
            ekuitas.reset_index(names='Date',inplace=True)
            ekuitas['Date'] = [i.strftime('%Y-%m') for i in ekuitas['Date']]
            json_ekuitas=ekuitas.to_json(orient='values')
            #ekuitas 2
            balance_sheet2 = ticker2.quarterly_balance_sheet
            ekuitas2 = balance_sheet2.loc[['Stockholders Equity','Retained Earnings','Common Stock']].transpose()
            ekuitas2.reset_index(names='Date',inplace=True)
            ekuitas2['Date'] = [i.strftime('%Y-%m') for i in ekuitas2['Date']]
            json_ekuitas2=ekuitas2.to_json(orient='values')

            return render_template('fundamental1.html', json_ekuitas=json_ekuitas, json_ekuitas2=json_ekuitas2, long_name=long_name2)
        
        return render_template('fundamental1.html', long_name=long_name, long_name2=long_name2, json_dividen=json_dividen, json_dividen2=json_dividen2, json_financial=json_financial, json_financial2=json_financial2, json_modal=json_modal, json_modal2=json_modal2, json_ekuitas=json_ekuitas, json_ekuitas2=json_ekuitas2)
    

    return render_template('fundamental1.html')


@app.route('/profil')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    app.run()