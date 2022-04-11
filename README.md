LSTM Training process

The deployment phase includes both open loop test and closed loop test.
In the open loop test there is a prediction one hour ahead using real data as new inputs.
In the closed loop test the previous prediction are used to generate new predictions.
To validate the closed loop it is necessary to perform an EnergyPlus simulations.
The first one use the same temperature predicted by the neural networks as a schedule
to obtain the amount of cooling power necessary to achieve these temperatures. As a result,
there is a comparison between the two temperatures and the two cooling power, that let us understand
what is the range of error of both temperature and cooling power and how they are related.