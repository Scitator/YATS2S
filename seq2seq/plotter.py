import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use("ggplot")
import seaborn as sns  # noqa: E402

sns.set(color_codes=True)


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_rutine(
        history, metric,
        legend=None, additional_info=None,
        show=False, save_dir=None,
        last_steps=None, last_steps_reference=None):
    title = 'model {}'.format(metric)
    if additional_info is not None:
        title += " {}".format(additional_info)
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')

    if save_dir is not None:
        filename = "{}/{}".format(save_dir, metric)
        if additional_info is not None:
            filename += "_{}".format(str(additional_info))
        filename += ".png"
        if legend is not None:
            plt.savefig(filename, format='png', dpi=300,
                        bbox_extra_artists=(legend,), bbox_inches='tight')
        else:
            plt.savefig(filename, format='png', dpi=300)
    if show:
        plt.show()

    if last_steps is not None and isinstance(last_steps, float):
        last_steps = int(last_steps * len(history[metric]))

    if last_steps is not None and isinstance(last_steps, int) and last_steps_reference is not None:
        last_steps_history = {metric: history[metric][-last_steps:]}
        val_metric = "val_{}".format(metric)
        if val_metric in history.keys():
            last_steps_history[val_metric] = history[val_metric][-last_steps:]
        last_steps_reference(
            history=last_steps_history,
            metric=metric,
            additional_info="last_steps",
            show=show,
            save_dir=save_dir)


def plot_bimetric(
        history, metric,
        additional_info=None,
        show=False, save_dir=None, last_steps=None):
    plt.figure()

    if "log_epochs" in history:
        plt.plot(history["log_epochs"], history[metric])
        plt.plot(history["log_epochs"], history['val_{}'.format(metric)])
    else:
        plt.plot(history[metric])
        plt.plot(history['val_{}'.format(metric)])

    lgn = plt.legend(['train', 'val'], loc='center left',
                     bbox_to_anchor=(1, 0.5))

    plot_rutine(
        history, metric,
        legend=lgn, additional_info=additional_info,
        show=show, save_dir=save_dir,
        last_steps=last_steps, last_steps_reference=plot_bimetric)


def plot_unimetric(
        history, metric,
        additional_info=None,
        show=False, save_dir=None, last_steps=None):
    plt.figure()

    if "log_epochs" in history:
        plt.plot(history["log_epochs"], history[metric])
    else:
        plt.plot(history[metric])

    plot_rutine(
        history, metric,
        legend=None, additional_info=additional_info,
        show=show, save_dir=save_dir,
        last_steps=last_steps, last_steps_reference=plot_unimetric)


def plot_all_metrics(
        history,
        show=False, save_dir=None, last_steps=None):
    if save_dir is not None:
        create_if_need(save_dir)

    bimetrics = []
    for metric in history.keys():
        if metric.startswith("val_"):
            bimetrics.append(metric[4:])

    for metric in history.keys():
        if not metric.startswith("val_"):
            if metric in bimetrics:
                plot_bimetric(
                    history, metric,
                    additional_info=None, show=show,
                    save_dir=save_dir, last_steps=last_steps)
            else:
                plot_unimetric(
                    history, metric,
                    additional_info=None, show=show,
                    save_dir=save_dir, last_steps=last_steps)
