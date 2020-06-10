from keras import callbacks

checkpoints = [x * 10 for x in range(10)] + [x * 10 for x in range(10, 20, 2)] + [x * 10 for x in range(20, 50, 5)]


class SaveOnEpochCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 100 == 0:  # or save after some epoch, each 100-th epoch etc.
            try:
                self.model.save("./history/model_{}.hd5".format(epoch))
                return
            except:
                print("NO SPACE LEFT ON DRIVE")
                pass

        if epoch % 50 == 0:  # or save after some epoch, each 50-th epoch etc.
            try:
                self.model.save("../../../../history/model_{}.hd5".format(epoch))
            except:
                print("NO SPACE LEFT ON COLABS")
