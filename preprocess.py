import numpy as np
import os
import pandas as pd



def preprocess_function(path,j,lookback):

    filelist = os.listdir(path)
    for file in filelist:
        df = pd.read_csv(path + file, engine='python')
        df1 = pd.DataFrame(columns=["Month", "DAY AND HOURS"])
        df1[["Month", "DAY AND HOURS"]] = df['Date/Time'].str.split("/", expand=True)
        df2 = pd.DataFrame(columns=['Days', 'HOURS'])
        df2[["Days", "HOURS"]] = df1['DAY AND HOURS'].str.split(expand=True)
        df1 = pd.concat([df1, df2], axis=1)
        df1.drop('DAY AND HOURS', axis=1, inplace=True)
        df = pd.concat([df1, df], axis=1)
        df3 = pd.DataFrame(columns=["Hour", "minutes", "seconds"])
        df3[["Hour", "minutes", "seconds"]] = df2['HOURS'].str.split(":", expand=True)
        df = pd.concat([df3, df], axis=1)
        df.drop('minutes', axis=1, inplace=True)
        df.drop('seconds', axis=1, inplace=True)
        df.drop('HOURS', axis=1, inplace=True)
        df['Month'] = df['Month'].astype(float)
        df['Hour'] = df['Hour'].astype(float)
        df['Days'] = df['Days'].astype(float)


        dfclean = df[['Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
                      'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)', 'Total People'
            , 'Mean air Temperature [C]', 'Total Cooling Rate [W]']]

        # =====================================
        # TRASFORMAZIONE DELLE VARIABILI TEMPORALI IN SENO E COSENO
        sinhour = np.sin(2 * np.pi * df['Hour'] / 24)
        coshour = np.cos(2 * np.pi * df['Hour'] / 24)

        sinday = np.sin(2 * np.pi * df['Environment:Site Day Type Index [](Hourly)'] / 7)
        cosday = np.cos(2 * np.pi * df['Environment:Site Day Type Index [](Hourly)'] / 7)

        sinmonth = np.sin(2 * np.pi * df['Month'] / 12)
        cosmonth = np.cos(2 * np.pi * df['Month'] / 12)


        column = ['Direct Solar Rad', 'T_ext', 'Occupants', 'T_int', 'Q_cooling']
        dfclean.columns = column
        Temp = dfclean.pop('T_int')
        type(Temp)

        dfclean = dfclean.assign(sinhour=sinhour.values)
        dfclean = dfclean.assign(coshour=coshour.values)
        dfclean = dfclean.assign(sinday=sinday.values)
        dfclean = dfclean.assign(cosday=cosday.values)
        dfclean = dfclean.assign(sinmonth=sinmonth.values)
        dfclean = dfclean.assign(cosmonth=cosmonth.values)
        dfclean = dfclean.assign(T_int=Temp.values)


        if j == 0:
            dftotal = dfclean
        else:
            dftotal = pd.concat([dftotal, dfclean], axis=0)
        j=j+1


    max = np.max(np.array(dftotal), axis=0).astype(np.float32)
    min = np.min(np.array(dftotal), axis=0).astype(np.float32)



    j=0
    for file in filelist:
        df = pd.read_csv(path+file, engine='python')
        ll = (len(df))
        inputs = np.zeros((len(df) - lookback - 1, lookback, 11))  # QUA HO DOVUTO INSERIRLO A MANO
        labels = np.zeros(len(df) - lookback - 1)

        df1 = pd.DataFrame(columns=["Month", "DAY AND HOURS"])
        df1[["Month", "DAY AND HOURS"]] = df['Date/Time'].str.split("/", expand=True)
        df2 = pd.DataFrame(columns=['Days', 'HOURS'])
        df2[["Days", "HOURS"]] = df1['DAY AND HOURS'].str.split(expand=True)
        df1 = pd.concat([df1, df2], axis=1)
        df1.drop('DAY AND HOURS', axis=1, inplace=True)
        df= pd.concat([df1, df], axis=1)
        df3 = pd.DataFrame(columns=["Hour", "minutes", "seconds"])
        df3[["Hour", "minutes", "seconds"]] = df2['HOURS'].str.split(":", expand=True)
        df = pd.concat([df3, df], axis=1)
        df.drop('minutes', axis=1, inplace=True)
        df.drop('seconds', axis=1, inplace=True)
        df.drop('HOURS', axis=1, inplace=True)
        df['Month'] = df['Month'].astype(float)
        df['Hour'] = df['Hour'].astype(float)
        df['Days'] = df['Days'].astype(float)

        dfclean = df[['Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
                      'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)', 'Total People'
            , 'Mean air Temperature [C]', 'Total Cooling Rate [W]']]


        # =====================================
        # Transform temporal variable in sine and cosine
        sinhour = np.sin(2 * np.pi * df['Hour'] / 24)
        coshour = np.cos(2 * np.pi * df['Hour'] / 24)

        sinday = np.sin(2 * np.pi * df['Environment:Site Day Type Index [](Hourly)'] / 7)
        cosday = np.cos(2 * np.pi * df['Environment:Site Day Type Index [](Hourly)'] / 7)

        sinmonth = np.sin(2 * np.pi * df['Month'] / 12)
        cosmonth = np.cos(2 * np.pi * df['Month'] / 12)

        column = ['Direct Solar Rad', 'T_ext', 'Occupants', 'T_int', 'Q_cooling']
        dfclean.columns = column
        Temp = dfclean.pop('T_int')
        type(Temp)

        dfclean = dfclean.assign(sinhour=sinhour.values)
        dfclean = dfclean.assign(coshour=coshour.values)
        dfclean = dfclean.assign(sinday=sinday.values)
        dfclean = dfclean.assign(cosday=cosday.values)
        dfclean = dfclean.assign(sinmonth=sinmonth.values)
        dfclean = dfclean.assign(cosmonth=cosmonth.values)
        dfclean = dfclean.assign(T_int=Temp.values)
        dfclean = dfclean.to_numpy().astype(np.float32)
        dfclean_scaled=np.zeros((len(dfclean),len(dfclean[:][0])))
        for z in range(0,len(dfclean)):
            num = np.subtract(dfclean[z,:],min)
            denom = np.subtract(max,min)
            dfclean_scaled[z,:]=np.divide(num,denom)


        lb = lookback+1

        for i in range(lb, len(df)):
            # inputs[i - lb] = dfclean_scaled[i - lookback:i,:-1]  # multivariate analysis
            Current_variables = dfclean_scaled[i - lookback:i, :-1]
            T_lag =dfclean_scaled[i-lb:(i-1), -1]
            inputs[i - lb,:,:] = np.column_stack([Current_variables, T_lag])
            labels[i - lb] = dfclean_scaled[i-1, -1]  # predict only the temperature that shuold be the LAST COLUMN

        if j==0:
            total_inputs=inputs
            total_labels=labels

        else:
            total_inputs = np.concatenate([total_inputs,inputs],axis=0)
            total_labels = np.concatenate([total_labels,labels],axis=0)

        j=j+1
    maxT=max[-1]
    minT=min[-1]

    return(total_inputs,total_labels,maxT,minT,j)

