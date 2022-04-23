import numpy

class WatchDog():
    r"""
    Check if to take early stop.

    Typical usage example::

        wd = WatchDog(10, 10)
        model.train()
        for e_idx in range(epoch_num):
            wd.add_epoch_loss()
            for b_idx, train_data in enumerate(dataloader):
                train_ret = model(train_data)
                # return batch loss sum
                loss = loss_func(train_data, train_ret)
                if wd.check_batch_loss(batch_loss / batch_size):
                    break
                loss.backward()
                optimizer.step()
                wd.cum_epoch_loss(batch_loss)
                print("batch information")
            if wd.toStop:
                optimizer.zero_grad()
                print("early stop information")
                break
            print("epoch information")
        model.eval()
    """
    def __init__(self, mov_avg_len: int = 10, ratio_max: int = 10):
        r"""
        Args:
            mov_avg_len: moving average length
            ratio_max: max ratio allowed of moving average to current loss
        """
        self.epoch_loss_arr = numpy.array([])  # epoch total loss
        self.batch_loss_arr = numpy.array([])  # batch average loss
        self.mov_avg_len = mov_avg_len  # for batches
        self.ratio_max = ratio_max  # max batch avg loss allowed
        self.ratio_this = 0  # batch avg loss this time
        self.toStop = False  # should stop training
        pass

    def __len__(self):
        """
        return epoch and batch loss array length
        """
        return len(self.epoch_loss_arr), len(self.batch_loss_arr)

    def reset(self, mov_avg_len: int = None):
        r"""
        reset before reuse
        """
        if mov_avg_len != None:
            self.mov_avg_len = mov_avg_len
        self.epoch_loss_arr = numpy.array([])  # epoch total loss
        self.batch_loss_arr = numpy.array([])  # batch average loss
        self.avg = 0
        self.ratio_this = 0  # batch avg loss this time
        self.toStop = False

    def check_batch_loss(self, loss_avg) -> bool:
        r"""
        Do a check if to stop training.

        Args:
            loss_avg: batch loss / batch item number

        Usage::

            if wd.check_batch_loss(batch_loss / batch_size):
                    break
        """
        if len(self.batch_loss_arr) > self.mov_avg_len:
            # ratio on exit is stored in self.ratio_this
            self.ratio_this = loss_avg / numpy.mean(self.batch_loss_arr[-self.mov_avg_len:])
            if self.ratio_this > self.ratio_max:  # check
                self.toStop = True
                # del the last epoch_loss
                self.epoch_loss_arr = self.epoch_loss_arr[:-1]
                return self.toStop
        self.batch_loss_arr = numpy.append(self.batch_loss_arr, loss_avg)
        return self.toStop

    def show_batch_loss_avg(self, n: int = None):
        r"""
        Show recent average item loss (not batch loss average)

        Args:
            n: number of items to show, self-adaptive if None
        """
        if type(n) is int:
            if n > 0 and n <= len(self.batch_loss_arr):
                return numpy.mean(self.batch_loss_arr[-n:])
            else:
                return numpy.mean(self.batch_loss_arr)
        else:
            return numpy.mean(self.batch_loss_arr[-self.mov_avg_len:])

    def add_epoch_loss(self):
        r"""
        Add a new position in self.epoch_loss_arr.

        Use before an epoch start.
        """
        self.epoch_loss_arr = numpy.append(self.epoch_loss_arr, 0)

    def cum_epoch_loss(self, batch_loss):
        r"""
        Cumulate batch loss to the last position in self.epoch_loss_arr.

        Args:
            batch_loss: batch loss (not divided by batch_size)
        """
        self.epoch_loss_arr[-1] += batch_loss

    def avg_epoch_loss(self, n):
        r"""
        Set the epoch loss to item average.

        n is usually the dataset size.

        Args:
            n: the divisor of the epoch loss, usually the dataset size.
        """
        self.epoch_loss_arr[-1] /= n
        return self.epoch_loss_arr[-1]
