import torch 








def _slid_win(series, win_size, look_back):
    """
    Generates sliding window input-output pairs from a time series.

    Parameters:
    - series (torch.Tensor): The time series data as a PyTorch tensor.
    - win_size (int): The size of the sliding window (number of time steps in the input sequence).
    - look_back (int): Number of most recent time steps to consider. If None, uses the entire series.

    Returns:
    - X (torch.Tensor): Input tensor of shape (num_windows, 1, win_size).
    - y (torch.Tensor): Target tensor of shape (num_windows, 1).
    """


    series = series.tolist()


    if look_back:
        series = series[-look_back:]

    X = []  
    y = []


    for i in range(len(series) - win_size):
        X.append(series[i:i + win_size])
        y.append(series[i + win_size])   


    X = torch.tensor(X).reshape(len(X), 1, win_size)


    y = torch.tensor(y).reshape(-1, 1)

    return X, y

