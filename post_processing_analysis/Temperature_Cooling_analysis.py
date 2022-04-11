import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Results_data_frame = pd.read_csv('post_processing_analysis/comparison_Tint_and_power.csv')


#===================================================#
#RESTAURANT
Restaurant_T_error = Results_data_frame[['ERR T RESTAURANT']].values
Restaurant_ideal_T = Results_data_frame[['IDEAL RESTAURANT']].values
Restaurant_predicted_T = Results_data_frame[['PREDICTED RESTAURANT']].values

Restaurant_Q_error = Results_data_frame[['ERR COOLING RESTAURANT']].values
Restaurant_ideal_Q = Results_data_frame[['IDEAL COOLING RESTAURANT']].values
Restaurant_predicted_Q = Results_data_frame[['PREDICTED COOLING RESTAURANT']].values




#===========================================================================
#PLOT
sns.set_color_codes()
plt.hist(Restaurant_Q_error, bins = 50, color ='orange',edgecolor = 'black' )
plt.xlim(left=-50,right=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Restaurant')
plt.ylabel('Count')
plt.xlabel('Cooling Power Relative Error')
plt.show()

plt.hist(Restaurant_T_error, bins = 30, color ='yellow',edgecolor = 'black' )
plt.xlim(left=-5,right=5)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title('Restaurant')
plt.ylabel('Count')
plt.xlabel(' Mean Indoor Temperature Relative Error')
plt.show()


labels = ["E+", "LSTM"]
plt.boxplot([Restaurant_ideal_T.flatten(),Restaurant_predicted_T.flatten()], labels = labels, showmeans=True, meanline=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.ylabel('Mean Indoor Temperature [°C]')
plt.title('Restaurant')
plt.show()


plt.boxplot([Restaurant_ideal_Q.flatten(),Restaurant_predicted_Q.flatten()], labels = labels, showmeans=True, meanline=True)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.05)
plt.ylabel('Total Cooling Power [W]')
plt.title('Restaurant')
plt.show()

plt.scatter(Restaurant_ideal_T,Restaurant_predicted_T,  color='yellow', edgecolor= 'white', linewidth=1,alpha=1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('E+ Mean Indoor Temperature [°C]')
plt.ylabel('LSTM Mean Indoor Temperature [°C]')
plt.title('Restaurant')
plt.show()



plt.scatter(Restaurant_ideal_Q,Restaurant_predicted_Q,  color='orange', edgecolor= 'white', linewidth=1,alpha=1)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Ideal Cooling Power [W]')
plt.ylabel('LSTM Cooling Power [W]')
plt.title('Restaurant')
plt.show()