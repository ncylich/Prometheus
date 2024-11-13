forecast_size = 36
backcast_size = forecast_size * 2

factor = 1
seq_len = backcast_size + forecast_size
nhid = 128 * factor
nhead = 8
dim_feedfwd = 512 * factor
nlayers = 12
dropout = 0.1
batch_size = 1024
test_col = 'close'

lr = 2e-4
epochs = 100
init_weight_magnitude = 1e-3