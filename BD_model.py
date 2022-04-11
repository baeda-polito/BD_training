import torch

#Creation of LSTM class this class is identical to the "BuildingDynamic Class in 3DEM"
class BuildingDynamics(torch.nn.Module):
    def __init__(self,n_features,seq_length, n_hidden,n_layers):
        super(BuildingDynamics, self).__init__()
        self.seq_len = seq_length
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(input_size = n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        # self.dropout = torch.nn.Dropout(drop_prob)

        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, x,h):
        batch_size, seq_len, _ = x.size()
        lstm_out, h = self.l_lstm(x,h)
        #out_numpy = lstm_out.detach().numpy()
        out = lstm_out[:,-1,:]  #many to one, I take only the last output vector, for each Batch
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        hidden = (hidden_state, cell_state) #HIDDEN is defined as a TUPLE
        return hidden

