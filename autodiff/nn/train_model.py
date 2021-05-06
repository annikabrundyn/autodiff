def train(
        self,
        X_train,
        Y_train,
        learning_rate,
        epochs,
        loss_function,
        verbose=False
):
    """Trains.

    Fits the model using the given parameters.

    Parameters
    ----------
    X_train : numpy.Array
        Training data. Must match the input size of the first layer.
    Y_train : numpy.Array
        Training labels.
    learning_rate : float
        Number of epochs to train the model
    epochs : int
        asdad
    loss_function : str
        Chosen function to compute loss.

    """
    for epoch in range(epochs):
        loss = self._run_epoch(X_train, Y_train, learning_rate, loss_function)

        if verbose:
            if epoch % 50 == 0:
                print(f'Epoch: {epoch}. Loss: {loss}')


def _run_epoch(self, X, Y, learning_rate, loss_function):
    """Runs epoch.

    Helper function of train procedure.

    Parameters
    ----------
    X_train : numpy.Array
        Training data. Must match the input size of the first layer.
    Y_train : numpy.Array
        Training labels.
    learning_rate : float
        Number of epochs to train the model
    epochs : int
        asdad
    loss_function : str
        Chosen function to compute loss.

    Returns
    -------
    error : float
        Model error in this epoch.

    """
    # Forward pass
    for i, _ in enumerate(self.layers):
        forward = self.layers[i].forward(input_val=X)
        X = forward

    # Compute loss and first gradient
    if loss_function == "BinaryCrossEntropy":
        loss_f = BinaryCrossEntropy(forward, Y)
    elif loss_function == "MeanSquaredError":
        loss_f = MeanSquaredError(forward, Y)
    else:
        raise ValueError(f"{loss_function} is not supported.")

    error = loss_f.forward()
    gradient = loss_f.backward()

    self.loss.append(error)

    # Backpropagation
    for i, _ in reversed(list(enumerate(self.layers))):
        if self.layers[i].type != 'Linear':
            gradient = self.layers[i].backward(gradient)
        else:
            gradient, dW, dB = self.layers[i].backward(gradient)
            self.layers[i].optimize(dW, dB, learning_rate)

    return error