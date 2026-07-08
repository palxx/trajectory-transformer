import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _append_csv_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


TRAINING_LOG_FIELDS = ['epoch', 'loss']

def log_training_loss(savepath, epoch, loss):
    '''
        Appends one row to `<savepath>/training_log.csv` so progress can be
        tailed or plotted while training is still running.
    '''
    path = os.path.join(savepath, 'training_log.csv')
    _append_csv_row(path, TRAINING_LOG_FIELDS, {'epoch': epoch, 'loss': loss})
    return path


def plot_training_curve(savepath):
    '''
        Reads `<savepath>/training_log.csv` and (re)writes
        `<savepath>/training_loss.png`.
    '''
    rows = _read_csv_rows(os.path.join(savepath, 'training_log.csv'))
    if not rows:
        return None

    epochs = [int(r['epoch']) for r in rows]
    losses = [float(r['loss']) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(epochs, losses, marker='o', markersize=3)
    ax.set_xlabel('epoch')
    ax.set_ylabel('train loss')
    ax.set_title('Training loss')
    ax.grid(alpha=0.3)

    plot_path = os.path.join(savepath, 'training_loss.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


EVAL_LOG_FIELDS = [
    'time', 'exp_name', 'suffix', 'gpt_loadpath', 'gpt_epoch',
    'target_return', 'return', 'score', 'step', 'term',
]

def log_evaluation_result(logbase, dataset, result):
    '''
        Appends one row (one evaluation episode) to
        `<logbase>/<dataset>/evaluation_log.csv`, so scores from every
        plan.py run on this dataset accumulate into one comparable file.
    '''
    path = os.path.join(logbase, dataset, 'evaluation_log.csv')
    row = {key: result.get(key) for key in EVAL_LOG_FIELDS}
    _append_csv_row(path, EVAL_LOG_FIELDS, row)
    return path


def plot_evaluation_scores(logbase, dataset):
    '''
        Reads `<logbase>/<dataset>/evaluation_log.csv` and (re)writes
        `<logbase>/<dataset>/evaluation_scores.png`: one point per evaluation
        episode, colored by `gpt_loadpath` so different trained models /
        runs are distinguishable on the same plot.
    '''
    path = os.path.join(logbase, dataset, 'evaluation_log.csv')
    rows = _read_csv_rows(path)
    if not rows:
        return None

    fig, ax = plt.subplots()

    run_names = sorted(set(r['gpt_loadpath'] for r in rows))
    cmap = plt.get_cmap('tab10')
    colors = {name: cmap(i % 10) for i, name in enumerate(run_names)}

    ## episode index within each run, so repeated episodes share an x-axis
    counters = {}
    for r in rows:
        name = r['gpt_loadpath']
        counters[name] = counters.get(name, -1) + 1
        ax.scatter(counters[name], float(r['score']), color=colors[name])

    ## proxy scatter points so each run gets exactly one legend entry
    for name in run_names:
        ax.scatter([], [], color=colors[name], label=name)

    ax.set_xlabel('episode')
    ax.set_ylabel('normalized score')
    ax.set_title(f'Evaluation score per episode ({dataset})')
    ax.legend(fontsize='small')
    ax.grid(alpha=0.3)

    plot_path = os.path.join(logbase, dataset, 'evaluation_scores.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def plot_target_vs_actual_return(logbase, dataset, gpt_loadpath):
    '''
        Reads `<logbase>/<dataset>/evaluation_log.csv`, filters to episodes
        evaluated against `gpt_loadpath`, and plots target return vs. actual
        achieved return for each one, against baseline (the y=x line where a
        perfectly-conditioned policy would land).
    '''
    path = os.path.join(logbase, dataset, 'evaluation_log.csv')
    rows = _read_csv_rows(path)
    rows = [r for r in rows if r['gpt_loadpath'] == gpt_loadpath]
    if not rows:
        return None

    rows = sorted(rows, key=lambda r: float(r['target_return']))
    targets = [float(r['target_return']) for r in rows]
    actuals = [float(r['return']) for r in rows]

    fig, ax = plt.subplots()

    lo, hi = min(targets + actuals), max(targets + actuals)
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='gray', label='baseline (target = actual)')

    ax.plot(targets, actuals, marker='o', color='tab:blue', label='actual return')
    ax.scatter(targets, targets, marker='x', color='tab:orange', label='target return')

    ax.set_xlabel('target return')
    ax.set_ylabel('return')
    ax.set_title(f'Target vs. actual return ({gpt_loadpath}, {dataset})')
    ax.legend(fontsize='small')
    ax.grid(alpha=0.3)

    plot_path = os.path.join(logbase, dataset, 'target_vs_actual_return.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path
