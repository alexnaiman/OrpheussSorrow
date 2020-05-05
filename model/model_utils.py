import keras

checkpoints = [x * 10 for x in range(10)] + [x * 10 for x in range(10, 20, 2)] + [x * 10 for x in range(20, 50, 5)]


class SaveOnEpochCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch in checkpoints or epoch % 2 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_{}.hd5".format(epoch))
