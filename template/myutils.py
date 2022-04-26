import numpy
from IPython import display
from matplotlib import pyplot as plt
import matplotlib_inline.backend_inline


# TODO: try_gpu()

class Animator:
    """
    For pretty display svg figures and running tips.
    Typical usage example::

        amt = Animator(1,2)
        x, y = getdata(...)
        amt.config_ax_(amt.axes[0])
        amt.plot_ax(amt.axes[0], x[:i], y[:,:i])
        amt.show()
    """

    def __init__(
        self, nrows: int = 1, ncols: int = 1, figsize: tuple = (4, 3),
    ):
        matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
        self.fig, self.axes = plt.subplots(
            nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows)
        )
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]  # make it iterable
        self.config_ax_hold = []  # record the config_ax settings
        self.adjust_space = plt.subplots_adjust

    @staticmethod
    def config_ax(
        ax,
        xlabel=None,
        ylabel=None,
        xlim: tuple = None,
        ylim: tuple = None,
        xscale: str = None,
        yscale: str = None,
        legend: list = None,
        title=None,
        grid=True,
    ):
        """
        func which do config the ax\n
        do not use directly
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)
        if legend is not None:
            ax.legend(legend)
        ax.grid(grid)

    def config_ax_(
        self,
        ax,
        xlabel=None,
        ylabel=None,
        xlim: tuple = None,
        ylim: tuple = None,
        xscale: str = "linear",
        yscale: str = "linear",
        legend: list = None,
        title=None,
        grid=True,
    ):
        """record the ax config"""
        self.config_ax_hold.append(
            lambda: self.config_ax(
                ax=ax,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                xscale=xscale,
                yscale=yscale,
                legend=legend,
                title=title,
                grid=grid,
            )
        )

    def config_axes(self):
        """run axes config"""
        for c in self.config_ax_hold:
            c()

    @staticmethod
    def plot_ax(ax, x, y, fmt=None, clear=True):
        """
        (x, y) = 1D, 1D
        (x, y) = 1D, 2D
        (x, y) = 2D, 2D and match size
        """
        if clear:
            ax.cla()
        if (not hasattr(x[0], "__len__")) and (not hasattr(y[0], "__len__")):
            # y is 1D
            if fmt is None:
                ax.plot(x, y)
            else:
                ax.plot(x, y, fmt)
        elif hasattr(y[0], "__len__"):
            if not hasattr(x[0], "__len__"):
                x = [x] * len(y)
            if fmt is None:
                for _x, _y in zip(x, y):
                    ax.plot(_x, _y)
            else:
                for _x, _y, _f in zip(x, y, fmt):
                    ax.plot(_x, _y, _f)
        else:
            print("plot error")

    def show(self):
        self.config_axes()
        display.display(self.fig),
        display.clear_output(wait=True)


class WatchLoss:
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
        if mov_avg_len is not None:
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
            self.ratio_this = loss_avg / numpy.mean(
                self.batch_loss_arr[-self.mov_avg_len:]
            )
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
