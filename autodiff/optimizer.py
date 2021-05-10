class SGD:

    def __init__(self, lr, params):
        self.lr = lr
        self.params = params

    def update_params(self, grads):
        #print('Update Params')
        for key in self.params:
            self.params[key] += - self.lr * grads['d' + key]

        return self.params