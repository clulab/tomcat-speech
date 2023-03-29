# plot the training and validation curves

import matplotlib.pyplot as plt


def plot_train_dev_curve(
    train_vals,
    dev_vals,
    x_label="",
    y_label="",
    title="",
    save_name=None,
    show=False,
    axis_boundaries=None,
):
    """
    plot the loss or accuracy/f1 curves over time for training and dev set
    :param train_vals: a list of losses/f1/acc on train set
    :param dev_vals: a list of losses/f1/acc on dev set
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param title: generic plot title; appended with loss or f1
    :param save_name: the name used for saving the image
    :param show: whether to show the image
    :param axis_boundaries: either (lower_bound, upper_bound) tuple
        or None, for automatic boundary setting
    """
    # get a list of the epochs
    epoch = [i for i, item in enumerate(train_vals)]

    # prepare figure
    fig, ax = plt.subplots()
    plt.grid(True)

    # add losses/epoch for train and dev set to plot
    ax.plot(epoch, train_vals, label="train")
    ax.plot(epoch, dev_vals, label="dev")

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # depending on type of input, set the y axis boundaries
    if axis_boundaries:
        ax.set_ylim(axis_boundaries[0], axis_boundaries[1])

    # create title and legend
    ax.set_title(title, loc="center", wrap=True)
    ax.legend()

    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()

    # show the plot
    if show:
        plt.show()


def plot_histograms_of_data_classes(
    data_list, x_label="", y_label="", title="", save_name="", show=False
):
    """
    Plot histograms for the number of items per class in the data
    :param data_list: a list containing all gold labels for dataset
    """
    # prepare figure
    fig, ax = plt.subplots()
    plt.grid(True)

    num_classes = set(data_list)

    # add losses/epoch for train and dev set to plot
    ax.hist(data_list, bins=[i for i in range(len(num_classes) + 1)])

    # label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # add title
    ax.set_title(title, loc="center", wrap=True)

    # save the file
    if save_name is not None:
        plt.savefig(fname=save_name)
        plt.close()

    # show the plot
    if show:
        plt.show()
