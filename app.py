import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)




var=st.sidebar.radio('select a option',('Info',"Holt-Winters",'LSTM','MULTIVARIATE',"FB PROPHET"))
if var=='Info':
    st.title("Stock Market Prediction Model App")

    st.header("Project Overview")
    st.write("""
    This project was developed by a team of students from ICT Mumbai as part of our Masters in Engineering Mathematics program. The goal of this project is to create an application that can predict stock market trends using advanced mathematical and machine learning techniques.
    """)

    st.header("Team Members")
    st.write("""
    - **Omkar**
    - **Ankit**
    - **Advay**
    - **Rukaiya**
    """)

    st.header("Features")
    st.write("""
    - **Data Collection**: Fetches real-time and historical stock market data.
    - **Data Preprocessing**: Cleans and preprocesses data for better accuracy.
    - **Prediction Models**: Implements various machine learning models for stock price prediction.
    - **Visualization**: Provides graphical representation of stock trends and predictions.
    - **User Interface**: Easy-to-use interface for users to interact with the prediction models.
    """)

    st.header("Technology Stack")
    st.write("""
    - **Programming Languages**: Python
    - **Frameworks and Libraries**: 
      - Streamlit (for the web app)
      - Scikit-learn (for machine learning models)
      - Pandas (for data manipulation)
      - Matplotlib/Plotly (for data visualization)
      - TensorFlow/Keras (for deep learning models)
    - **APIs**: Yahoo Finance API (for stock data)
    - **Version Control**: Git and GitHub
    """)

    st.header("Contact")
    st.write("""
    For any queries or suggestions, please reach out to us:
    - **Omkar**: [mat22ot.omkar@pg.ictmumbai.edu.in]
    - **Ankit**: [mat22ag.rai@pg.ictmumbai.edu.in]
    - **Advay**: [mat22ak.parab@pg.ictmumbai.edu.in]
    - **Rukaiya**: [mat22ra.shaikh@pg.ictmumbai.edu.in]
    """)
    
if var=="Holt-Winters":
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.api import ExponentialSmoothing
    import yfinance as yf

    tic = st.text_input('Enter symbol of the stock! you can find symbol of corresponding stock on https://finance.yahoo.com/')
    target_column = st.selectbox('select a variable',("High",'Close','Open','Low'))
    output_lenght = st.number_input('Number of days to forecast',
                                       min_value=1, max_value=15, value=1, step=1)
    submit = st.button('Submit')

    if submit==True:
        try:
            Getdata = yf.Ticker(tic)
            data=pd.DataFrame(Getdata.history(period="5y"))
            data.reset_index(inplace=True)
        except:
            st.text('please enter a valid symbol')
        test_size=60
        train=data[target_column][:-test_size]
        test=data[target_column][-test_size:]
        err=[]
        il=[]
        for i in range(10,260,10): 
            triple = ExponentialSmoothing(train,
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=240).fit(optimized=True)
            forecast_values= triple.forecast(test_size)
            err.append(np.mean(test-forecast_values)**2)
            il.append(i)
        sp=il[np.argmin(err)]
        triple = ExponentialSmoothing(data[target_column],
                                trend='add',
                                seasonal='add',
                                seasonal_periods=sp).fit(optimized=True)
        forecast_values= triple.forecast(output_lenght)
        st.text(np.array(forecast_values))


if var=='LSTM':
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout ,GRU, Input, Attention, Concatenate
    from keras.optimizers import Adam
    import tensorflow as tf
    from tensorflow.keras.models import Model
    import yfinance as yf

    tic = st.text_input('Enter symbol of the stock! you can find symbol of corresponding stock on https://finance.yahoo.com/')
    target_column = st.selectbox('select a variable',("High",'Close','Open','Low'))
    output_lenght = st.number_input('Number of days to forecast',
                                       min_value=1, max_value=15, value=1, step=1)
    sequence_length = st.number_input('Number of time steps used for forecasting',
                                            min_value=5, max_value=250, value=20, step=5)
    submit = st.button('Submit')

    if submit==True:
        try:
            Getdata = yf.Ticker(tic)
            data=pd.DataFrame(Getdata.history(period="5y"))
            data.reset_index(inplace=True)
        except:
            st.text('please enter a valid symbol')

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[target_column].values.reshape(-1, 1))

        # Create sequences of data
        sequences = []
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i : i + sequence_length + 1])

        # Convert sequences to numpy array
        sequences = np.array(sequences)

        # Split sequences into input (X) and output (y)
        X = sequences[:, :-output_lenght]
        y = sequences[:, -output_lenght:]

        temp= sequences[:,output_lenght:]


        def build_attention_lstm_model(input_shape, output_shape):
            # Define input layer
            inputs = Input(shape=input_shape)
            # Apply LSTM layer
            lstm_output = LSTM(60, return_sequences=True)(inputs)

            # Apply attention mechanism
            attention = Attention()([lstm_output, lstm_output])

            # Concatenate attention output with LSTM output
            concat = Concatenate(axis=-1)([lstm_output, attention])

            # Apply another LSTM layer
            lstm_output_2 = LSTM(50,return_sequences=True)(concat)
            lstm_output_3 = LSTM(30)(lstm_output_2)
            # Output layer
            Dence_output_1 = Dense(10, activation='relu')(lstm_output_3)
            outputs = Dense(output_shape, activation='linear')(Dence_output_1)

            # Define model
            model = Model(inputs=inputs, outputs=outputs)

            return model

        # Example usage
        input_shape = (sequence_length-output_lenght+1, 1)  # Assuming input time series of length 10 with 5 features
        output_shape = output_lenght # Assuming a single output value
        model = build_attention_lstm_model(input_shape, output_shape)

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Compile model
        #model.compile(optimizer="adam", loss="mean_squared_error")

        # Train model
        model.fit(X, y, epochs=50, batch_size=64,verbose=0)

        predictions = model.predict(temp)

        # Inverse transform predictions and actual values
        predictions = scaler.inverse_transform(predictions)
        st.text(predictions[-1])
    

if var=='MULTIVARIATE':
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout ,GRU, Input, Attention, Concatenate
    from keras.optimizers import Adam
    import tensorflow as tf
    from tensorflow.keras.models import Model
    import yfinance as yf

    tic = st.text_input('Enter symbol of the stock! you can find symbol of corresponding stock on https://finance.yahoo.com/')
    sequence_length = st.number_input('Number of time steps used for forecasting',
                                            min_value=20, max_value=250, value=20, step=5)

    submit = st.button('Submit')

    if submit==True:
        try:
            Getdata = yf.Ticker(tic)
            data=pd.DataFrame(Getdata.history(period="5y"))
            data.reset_index(inplace=True)
        except:
            st.text('please enter a valid symbol')

    
        # Choose the columns to predict
        target_columns = ["High", "Low", "Open", "Close", "Volume"]

        # Feature scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[target_columns])

        # Define sequence length (number of time steps to look back)
        
        # Create sequences of data
        sequences = []
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i : i + sequence_length + 1])

        sequences = np.array(sequences)

        # Split sequences into input (X) and output (y)
        X = sequences[:, :-1]
        y = sequences[:, -1]

        temp= sequences[:,1:]


        def build_attention_lstm_model(input_shape, output_shape):
            # Define input layer
            inputs = Input(shape=input_shape)
            # Apply LSTM layer
            lstm_output = LSTM(60, return_sequences=True)(inputs)

            # Apply attention mechanism
            attention = Attention()([lstm_output, lstm_output])

            # Concatenate attention output with LSTM output
            concat = Concatenate(axis=-1)([lstm_output, attention])

            # Apply another LSTM layer
            lstm_output_2 = LSTM(50,return_sequences=True)(concat)
            lstm_output_3 = LSTM(30)(lstm_output_2)
            # Output layer
            Dence_output_1 = Dense(10, activation='relu')(lstm_output_3)
            outputs = Dense(output_shape, activation='linear')(Dence_output_1)

            # Define model
            model = Model(inputs=inputs, outputs=outputs)

            return model

        # Example usage
        input_shape = (sequence_length, 5)  # Assuming input time series of length 10 with 5 features
        output_shape = 5  # Assuming a single output value
        model = build_attention_lstm_model(input_shape, output_shape)

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Compile model
        #model.compile(optimizer="adam", loss="mean_squared_error")

        # Train model
        model.fit(X, y, epochs=50, batch_size=64,verbose=0)

        predictions = model.predict(temp)

        # Inverse transform predictions and actual values
        predictions = scaler.inverse_transform(predictions)
        st.text(f"High:{predictions[-1,0]}, Low:{predictions[-1][1]}, Open:{predictions[-1][2]}, Close:{predictions[-1][3]}")
    
if var=="FB PROPHET":
    import pandas as pd
    from prophet.plot import plot_plotly, plot_components_plotly
    from prophet import Prophet
    import matplotlib.pyplot as plt
    import yfinance as yf

    tic = st.text_input('Enter symbol of the stock! you can find symbol of corresponding stock on https://finance.yahoo.com/')
    target_column = st.selectbox('select a variable',("High",'Close','Open','Low'))
    output_lenght = st.number_input('Number of days to forecast',
                                        min_value=20, max_value=300, value=120, step=10)
    submit = st.button('Submit')

    if submit==True:
        try:
            Getdata = yf.Ticker(tic)
            data=pd.DataFrame(Getdata.history(period="5y"))
            data.reset_index(inplace=True)
        except:
            st.text('please enter a valid symbol')

        data['Date']=data['Date'].dt.strftime('%d-%m-%Y')

        # Choose the columns to predict
        data= data[["Date",target_column]]
        data.columns = ['ds','y']

        model=Prophet()
        model.fit(df=data)

        future = model.make_future_dataframe(periods=output_lenght,freq='D')
        forecast= model.predict(future)

        forecast_t = forecast['yhat'].values
        st.text(forecast_t[-output_lenght:])

        st.plotly_chart(plot_plotly(model,forecast))
