#%%
### NCF-t1: PROXIMAL TRAINING AND CONTEXT ANIMATIONS ###
# All experiment settings are below; this script takes no command-line arguments.
# train=True: create a new experiment beneath this example's runs/ directory.
# train=False: restore run_folder and regenerate figures and all six animations.
# Set run_folder=None to select the latest completed run when train=False.
# A copy of this script is saved in each run: set train=False there to reload it.
# Dependencies: the usual NCF environment, scikit-learn, and ffmpeg (GIF fallback).

from pathlib import Path
import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ncf import *

## Experiment settings ##
seed = 2026
train = True
run_folder = None                   ## None, or e.g. "./runs/260928-231433/"
save_trainer = True                 ## Required to reload later with train=False

## NCF main hyperparameters (same architecture and loss as main.py) ##
context_pool_size = 6
context_size = 1024
ncf_variant = 1                     ## NCF-t1: first-order Taylor expansion
nb_outer_steps_max = 4000           ## Record contexts after EVERY outer epoch
nb_inner_steps_max = 25
proximal_beta = 1e2
inner_tol_node = 1e-12
inner_tol_ctx = 1e-12
early_stopping_patience = None      ## Complete all 2000 outer epochs

## General training hyperparameters ##
print_error_every = 10
integrator = RK4
ivp_args = {"dt_init": 1e-4, "rtol": 1e-3, "atol": 1e-6, "max_steps": 40000, "subdivisions": 5}
init_lr = 1e-4
sched_factor = 1.0

## Adaptation hyperparameters ##
adapt_test = True
adapt_restore = not train          ## train=False restores adapted contexts too
sequential_adapt = True
init_lr_adapt = 1e-4
sched_factor_adapt = 1.0
nb_epochs_adapt = 2500

## Visualisation settings (editable when train=False, without retraining) ##
animate_contexts = True
animation_max_frames = 201         ## Movie subsampling only; all contexts are saved
animation_fps = 25
probe_relative_ridge = 1e-6        ## Fixed penalty; fitted only to 9 training targets
# The affine 1024 -> 2 probe is underdetermined. Held-out parameter error matters;
# a near-perfect fit to nine training points does not establish identifiability.

## Paths work from the example, repository root, or a saved run's script ##
script_path = Path(__file__).resolve()
example_folder = next((p for p in script_path.parents if (p / "dataset.py").is_file()), None)
if example_folder is None:
    raise FileNotFoundError("Keep this script in lotka-volterra/ or one of its runs/ folders")
runs_folder = example_folder / "runs"
data_folder = str(example_folder / "data") + "/"
# Edit data_folder above if using another dataset with dataset.py's environment order.


#%%
### CONTEXT RECORDING, LINEAR PROBE, AND ANIMATION HELPERS ###
# Exact environment order from dataset.py: columns are beta, delta.
TRAIN_PARAMETERS = np.array([
    [0.5, 0.5], [0.75, 0.5], [1.0, 0.5],
    [0.5, 0.75], [0.75, 0.75], [1.0, 0.75],
    [0.5, 1.0], [0.75, 1.0], [1.0, 1.0],
])
ADAPT_PARAMETERS = np.array([
    [0.625, 0.625], [0.625, 1.125], [1.125, 0.625], [1.125, 1.125],
])


class ContextHistory:
    """Flush each snapshot to a .npy memmap, including initialization at epoch 0."""

    def __init__(self, path, epochs, nb_envs, context_size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite context history: {self.path}")
        self.values = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=np.float32,
            shape=(epochs + 1, nb_envs, context_size),
        )
        self.values[:] = np.nan  # Unrecorded epochs remain distinguishable after interruption.
        self.values.flush()

    def __call__(self, epoch, params, env_id=None):
        values = np.asarray(params, dtype=np.float32)
        expected = self.values.shape[1:] if env_id is None else (1, self.values.shape[2])
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"Invalid contexts at epoch {epoch}, environment {env_id}")
        if env_id is None:
            self.values[epoch] = values
        else:
            self.values[epoch, env_id] = values[0]
        self.values.flush()


def load_history(path):
    values = np.load(path, mmap_mode="r")
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Incomplete or invalid context history: {path}")
    return values


def fit_linear_probe(contexts, targets, relative_ridge=1e-6):
    """Fit a centered affine ridge probe in the 9-sample dual space.

    The tiny fixed regularizer stabilizes the underdetermined 1024 -> 2 map.
    No adaptation contexts or labels enter the fit or hyperparameter selection.
    Training fit error alone is not evidence of physical identifiability.
    """
    x = np.asarray(contexts, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0], 2):
        raise ValueError("Expected one (beta, delta) target per training context")
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    gram = xc @ xc.T
    ridge = relative_ridge * max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    weights = xc.T @ np.linalg.solve(gram + ridge * np.eye(len(x)), yc)
    intercept = y_mean - x_mean @ weights
    return weights, intercept, ridge


def frame_indices(length, max_frames):
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    return np.unique(np.linspace(0, length - 1, min(length, max_frames), dtype=int))


def _limits(ax, arrays):
    points = np.concatenate([np.asarray(a).reshape(-1, 2) for a in arrays])
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = np.maximum(hi - lo, 0.1) * 0.12
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def _truth_grid(ax, targets, color, label):
    ax.scatter(*targets.T, c=color, alpha=0.45, s=35, label=label, zorder=2)
    for axis in (0, 1):
        for value in np.unique(targets[:, axis]):
            line = targets[targets[:, axis] == value]
            line = line[np.argsort(line[:, 1 - axis])]
            ax.plot(*line.T, color=color, alpha=0.3, lw=0.8)
    for i, point in enumerate(targets):
        ax.annotate(str(i), point, xytext=(-9, -11), textcoords="offset points",
                    color=color, fontsize=8)


def save_movie(coords, epochs, path, title, labels, reference=None,
               physical=False, adaptation=False, fps=None):
    """Editorial scientific layout with fading trails and a fixed coordinate frame."""
    fps = animation_fps if fps is None else fps
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ink, muted, paper = "#202b3a", "#6e7a89", "#fafaf7"
    amber, violet = "#de910a", "#8647bc"
    accent = violet if adaptation else amber
    palette = ["#de910a", "#37899b", "#8647bc", "#d56765", "#589c78",
               "#566bc3", "#aa668b", "#7b8840", "#a16b43"]
    colors = [accent] * coords.shape[1] if physical else palette[:coords.shape[1]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted}):
        fig = plt.figure(figsize=(10, 7.2), facecolor=paper)
        ax = fig.add_axes([0.09, 0.22, 0.59, 0.58], facecolor="white")
        fig.text(0.085, 0.94, "NEURAL CONTEXT FLOW", fontsize=10, weight="bold",
                 color=accent, va="center")
        fig.text(0.915, 0.94, "LOTKA–VOLTERRA  /  NCF–t1", fontsize=9,
                 color=muted, ha="right", va="center")
        heading = "Recovering the physics" if physical else "The geometry of context"
        fig.text(0.085, 0.867, heading, fontsize=25, weight="bold")
        method = "Fixed linear probe  ·  β and δ" if physical else (
            "Fixed principal components" if "PCA" in title else "Joint t-SNE embedding")
        fig.text(0.085, 0.825, method, fontsize=11, color=muted)
        stage = "ADAPTATION" if adaptation else "META-TRAINING"
        fig.text(0.74, 0.74, stage, color=accent, fontsize=10, weight="bold")
        fig.text(0.74, 0.69, "LOCAL EPOCH" if adaptation else "OUTER EPOCH",
                 color=muted, fontsize=8, weight="bold")
        epoch_text = fig.text(0.74, 0.638, "", fontsize=25, weight="bold")
        fig.text(0.74, 0.605, f"of {epochs[-1]:,}", color=muted, fontsize=10)
        metric_label = "PARAMETER MSE" if physical else "ENVIRONMENTS"
        fig.text(0.74, 0.53, metric_label, color=muted, fontsize=8, weight="bold")
        metric_text = fig.text(0.74, 0.48, "", fontsize=19, weight="bold", color=accent)
        note = ("Probe fitted on\n9 final training contexts.\nHeld fixed throughout."
                if physical else "One coordinate frame\nfor the entire trajectory.")
        if adaptation:
            note += "\n\nEqual local epochs\nacross environments."
        if "t-SNE" in title:
            note += "\n\nAxes are not physical."
        fig.text(0.74, 0.38, note, fontsize=9, color=muted, linespacing=1.65, va="top")

        arrays = [coords]
        if physical:
            _truth_grid(ax, TRAIN_PARAMETERS, amber, "Train ground truth")
            arrays.append(TRAIN_PARAMETERS)
            if adaptation:
                _truth_grid(ax, ADAPT_PARAMETERS, violet, "Adapt ground truth")
                arrays.append(ADAPT_PARAMETERS)
        if reference is not None:
            ax.scatter(*reference.T, marker="X" if physical else "o",
                       color=amber, edgecolor="white", linewidth=0.6,
                       s=65, zorder=4)
            arrays.append(reference)
        halo = ax.scatter(*coords[0].T, c=colors, s=260, alpha=0.10,
                          edgecolors="none", zorder=3)
        points = ax.scatter(*coords[0].T, c=colors, marker="X" if physical else "o",
                            edgecolors="white", linewidths=0.9, s=90, zorder=6)
        trails = [LineCollection([], linewidths=2.2, zorder=3)
                  for _ in range(coords.shape[1])]
        for line in trails:
            ax.add_collection(line)
        texts = [ax.annotate(str(i), coords[0, i], xytext=(6, 7),
                            textcoords="offset points", fontsize=8, weight="bold",
                            color=colors[i], zorder=7)
                 for i in range(coords.shape[1])]
        _limits(ax, arrays)
        ax.set(xlabel=labels[0], ylabel=labels[1])
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(color="#e8ebed", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#dde2e6")
            spine.set_linewidth(0.8)

        legend = [
            Line2D([], [], marker="X" if physical else "o", color="none",
                   markerfacecolor=accent, markeredgecolor="white", markersize=9,
                   label="Adapt prediction" if physical and adaptation else
                   "Train prediction" if physical else "Current context"),
        ]
        if physical:
            legend.append(Line2D([], [], marker="o", color=accent, alpha=0.5,
                          linewidth=0.8, markersize=4, label="Ground truth grid"))
        if reference is not None:
            legend.append(Line2D([], [], marker="X" if physical else "o",
                          color="none", markerfacecolor=amber,
                          markeredgecolor="white", markersize=9, label="Final training"))
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.085, 0.10),
                   ncol=len(legend), frameon=False, fontsize=9,
                   handletextpad=0.5, columnspacing=1.5)
        timeline = fig.add_axes([0.09, 0.075, 0.825, 0.007])
        timeline.set(xlim=(0, 1), ylim=(0, 1))
        timeline.axis("off")
        timeline.axhline(0.5, color="#e0e4e7", linewidth=4)
        progress, = timeline.plot([], [], color=accent, linewidth=4, solid_capstyle="round")
        fig.text(0.09, 0.042, "INITIALISATION", fontsize=7, color=muted)
        fig.text(0.915, 0.042, "FINAL CONTEXT", fontsize=7, color=muted, ha="right")

        def update(i):
            points.set_offsets(coords[i])
            halo.set_offsets(coords[i])
            for env, (line, text) in enumerate(zip(trails, texts)):
                tail = coords[max(0, i - 35):i + 1, env]
                segments = np.stack([tail[:-1], tail[1:]], axis=1)
                line.set_segments(segments)
                rgba = np.tile(to_rgba(colors[env]), (len(segments), 1))
                rgba[:, 3] = np.linspace(0.03, 0.7, len(segments))
                line.set_color(rgba)
                text.xy = coords[i, env]
            epoch_text.set_text(f"{epochs[i]:,}")
            if physical:
                targets = ADAPT_PARAMETERS if adaptation else TRAIN_PARAMETERS
                metric_text.set_text(f"{np.mean((coords[i] - targets)**2):.2e}")
            else:
                metric_text.set_text(f"{coords.shape[1]:02d}")
            progress.set_data([0, (i / max(len(coords) - 1, 1))], [0.5, 0.5])
            return points, halo, progress, epoch_text, metric_text, *trails, *texts

        update(len(coords) - 1)
        fig.savefig(Path(path).with_suffix(".png"), dpi=220, facecolor=paper)
        animation = FuncAnimation(fig, update, frames=len(coords), interval=1000 / fps,
                                  blit=False)
        if writers.is_available("ffmpeg"):
            output = Path(path).with_suffix(".mp4")
            animation.save(output, writer=FFMpegWriter(
                fps=fps, codec="libx264", bitrate=3500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
                dpi=150, savefig_kwargs={"facecolor": paper})
        else:
            output = Path(path).with_suffix(".gif")
            print("ffmpeg unavailable; saving GIF instead:", output)
            animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
        plt.close(fig)
        print("Saved", output, flush=True)


def visualise_contexts(folder, train_history, final_train, adapt_history=None,
                       max_frames=201, seed=2026, sequential=True):
    """Fit once, save coordinates/probe, then render training and adaptation."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if train_history.shape[1] != len(TRAIN_PARAMETERS):
        raise ValueError("This example expects dataset.py's nine training environments")
    if adapt_history is not None and adapt_history.shape[1] != len(ADAPT_PARAMETERS):
        raise ValueError("This example expects dataset.py's four adaptation environments")
    if not np.allclose(train_history[-1], final_train):
        raise ValueError("Probe contexts must match the final recorded training epoch")

    weights, intercept, ridge = fit_linear_probe(final_train, TRAIN_PARAMETERS, relative_ridge=probe_relative_ridge)
    train_physical = train_history @ weights + intercept
    adapt_physical = None if adapt_history is None else adapt_history @ weights + intercept
    np.savez(folder / "linear_probe.npz", weights=weights, intercept=intercept,
             ridge=ridge, train_contexts=final_train, targets=TRAIN_PARAMETERS,
             parameter_names=np.array(["beta", "delta"]))
    np.savez(folder / "physical_coordinates.npz",
             train=train_physical, train_epochs=np.arange(len(train_history)),
             adapt=np.empty((0, 4, 2)) if adapt_physical is None else adapt_physical,
             adapt_epochs=np.arange(0 if adapt_history is None else len(adapt_history)),
             train_targets=TRAIN_PARAMETERS, adapt_targets=ADAPT_PARAMETERS)
    print(f"Probe training parameter MSE: {np.mean((train_physical[-1] - TRAIN_PARAMETERS)**2):.3e}")
    if adapt_physical is not None:
        print(f"Probe held-out adaptation parameter MSE: {np.mean((adapt_physical[-1] - ADAPT_PARAMETERS)**2):.3e}")

    train_epochs = frame_indices(len(train_history), max_frames)
    adapt_epochs = None if adapt_history is None else frame_indices(len(adapt_history), max_frames)
    adapt_title = "Adaptation (local epoch per environment)" if sequential else "Adaptation"
    for stage, coords, epochs in [
        ("train", train_physical[train_epochs], train_epochs),
        ("adapt", None if adapt_physical is None else adapt_physical[adapt_epochs], adapt_epochs),
    ]:
        if coords is not None:
            save_movie(coords, epochs, folder / f"{stage}_physical",
                       "NCF-t1: " + ("Training" if stage == "train" else adapt_title),
                       (r"$\beta$", r"$\delta$"), physical=True, adaptation=stage == "adapt",
                       reference=train_physical[-1] if stage == "adapt" else None)

    # PCA is fitted only on the final training contexts and reused at every epoch.
    pca = PCA(n_components=2, svd_solver="full").fit(final_train)
    def project(values):
        return pca.transform(values.reshape(-1, values.shape[-1])).reshape(*values.shape[:2], 2)
    train_pca = project(train_history)
    adapt_pca = None if adapt_history is None else project(adapt_history)
    np.savez(folder / "pca_coordinates.npz", train=train_pca,
             adapt=np.empty((0, 4, 2)) if adapt_pca is None else adapt_pca,
             mean=pca.mean_, components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_,
             train_epochs=np.arange(len(train_history)),
             adapt_epochs=np.arange(0 if adapt_history is None else len(adapt_history)))
    save_movie(train_pca[train_epochs], train_epochs, folder / "train_pca",
               "NCF-t1: Training — fixed PCA", ("PC 1", "PC 2"))
    if adapt_pca is not None:
        save_movie(adapt_pca[adapt_epochs], adapt_epochs, folder / "adapt_pca",
                   f"NCF-t1: {adapt_title} — fixed PCA", ("PC 1", "PC 2"),
                   reference=pca.transform(final_train), adaptation=True)

    # One joint t-SNE embedding, never independently re-fitted at each epoch.
    # This is a retrospective visualisation; t-SNE distances are not physical.
    train_sample = np.asarray(train_history[train_epochs])
    blocks = [train_sample.reshape(-1, train_sample.shape[-1])]
    if adapt_history is not None:
        adapt_sample = np.asarray(adapt_history[adapt_epochs])
        blocks.append(adapt_sample.reshape(-1, adapt_sample.shape[-1]))
    joint = np.concatenate(blocks)
    reduced = PCA(n_components=min(50, *joint.shape), random_state=seed).fit_transform(joint)
    embedded = TSNE(n_components=2, perplexity=min(30.0, (len(joint) - 1) / 3),
                    init="pca", learning_rate="auto", random_state=seed).fit_transform(reduced)
    split = len(blocks[0])
    train_tsne = embedded[:split].reshape(len(train_epochs), len(TRAIN_PARAMETERS), 2)
    adapt_tsne = None if adapt_history is None else embedded[split:].reshape(
        len(adapt_epochs), len(ADAPT_PARAMETERS), 2)
    np.savez(folder / "tsne_coordinates.npz", train=train_tsne,
             adapt=np.empty((0, 4, 2)) if adapt_tsne is None else adapt_tsne,
             train_epochs=train_epochs,
             adapt_epochs=np.array([], dtype=int) if adapt_epochs is None else adapt_epochs)
    save_movie(train_tsne, train_epochs, folder / "train_tsne",
               "NCF-t1: Training — joint t-SNE", ("t-SNE 1", "t-SNE 2"))
    if adapt_tsne is not None:
        save_movie(adapt_tsne, adapt_epochs, folder / "adapt_tsne",
                   f"NCF-t1: {adapt_title} — joint t-SNE", ("t-SNE 1", "t-SNE 2"),
                   reference=train_tsne[-1], adaptation=True)


#%%
class ContextRecordingTrainer(Trainer):
    """The original trainer loops with local, host-side context observers.

    Keeping these methods here leaves ncf/trainer.py and other examples untouched.
    The final model is saved alongside its matching final context snapshot.
    """

    def train_proximal(self, 
                       nb_outer_steps_max, 
                       int_prop=1.0, 
                       inner_tol_node=1e-2, 
                       inner_tol_ctx=1e-2, 
                       nb_inner_steps_max=10, 
                       proximal_reg=100., 
                       patience=None, 
                       print_error_every=1, 
                       save_path=False, 
                       val_dataloader=None, 
                       val_criterion=None, 
                       key=None,
                       context_callback=None):
        """ Train the NCF using the Proximal Alternating Mimisation algorithm. Algorithm 2 in https://proceedings.mlr.press/v97/li19n.html """

        # Observer receives (epoch, context_params, env_id=None) on the host.
        # Epoch 0 is initialization; other epochs follow both inner solves.
        key = key if key is not None else self.key

        opt_state_node = self.opt_node_state
        opt_state_ctx = self.opt_ctx_state

        loss_fn = self.learner.loss_fn

        node = self.learner.neuralode
        contexts = self.learner.contexts

        @eqx.filter_jit
        def train_step_node(node, node_old, contexts, batch, weights, opt_state, key):
            print('\nCompiling function "train_step" for neural ode ...')

            def prox_loss_fn(node, contexts, batch, weights, key):
                loss, aux_data = loss_fn(node, contexts, batch, weights, key)
                diff_norm = params_diff_norm_squared(node, node_old)
                return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(node, contexts, batch, weights, key)

            updates, opt_state = self.opt_node.update(grads, opt_state)
            node = eqx.apply_updates(node, updates)

            return node, contexts, opt_state, loss, aux_data

        @eqx.filter_jit
        def train_step_ctx(node, contexts, contexts_old, batch, weights, opt_state, key):
            print('\nCompiling function "train_step" for contexts ...')

            def prox_loss_fn(contexts, node, batch, weights, key):
                loss, aux_data = loss_fn(node, contexts, batch, weights, key)
                diff_norm = params_diff_norm_squared(contexts, contexts_old)
                return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, node, batch, weights, key)

            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return node, contexts, opt_state, loss, aux_data

        assert int_prop>0 and int_prop<=1.0, "The proportion of trajectory length to consider for training must be between 0 and 1"
        self.dataloader.int_cutoff = int(int_prop*self.dataloader.nb_steps_per_traj)

        if val_dataloader is not None:
            tester = VisualTester(self)

        print(f"\n\n=== Beginning training with proximal alternating minimization ... ===")
        print(f"    Number of examples in a batch: {self.dataloader.batch_size}")
        print(f"    Maximum number of steps per inner minimization: {nb_inner_steps_max}")
        print(f"    Maximum number of outer minimizations: {nb_outer_steps_max}")
        print(f"    Maximum total number of training steps: {nb_outer_steps_max*nb_inner_steps_max}")

        start_time = time.time()

        losses_node = []
        losses_ctx = []
        nb_steps_node = []
        nb_steps_ctx = []

        val_losses = []

        weights = jnp.ones(self.learner.nb_envs) / self.learner.nb_envs

        loss_key = get_new_key(key)

        early_stopping_count = 0

        if context_callback is not None:
            context_callback(0, contexts.params)

        for out_step in range(nb_outer_steps_max):

            node_old = jax.tree_util.tree_map(lambda x: x, node)
            contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

            ## Optimising the neural ode weights and biases
            node_prev = jax.tree_util.tree_map(lambda x: x, node)
            for in_step_node in range(nb_inner_steps_max):

                nb_batches_node = 0
                loss_sum_node = jnp.zeros(1)
                nb_steps_eph_node = 0

                for i, batch in enumerate(self.dataloader):
                    loss_key = get_new_key(loss_key)

                    node, contexts, opt_state_node, loss_node, (nb_steps_node_, term1, term2, diff_node_) = train_step_node(node, node_old, contexts, batch, weights, opt_state_node, loss_key)

                    loss_sum_node += jnp.array([loss_node])
                    nb_steps_eph_node += nb_steps_node_

                    nb_batches_node += 1

                diff_node = params_diff_norm_squared(node, node_prev) / params_norm_squared(node_prev)
                if diff_node < inner_tol_node or out_step==0:       ## Break early to see how big the loss is at the beginning
                    break
                node_prev = node

            loss_epoch_node = loss_sum_node/nb_batches_node

            ## Optimising the context parameters
            contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
            for in_step_ctx in range(nb_inner_steps_max):

                nb_batches_ctx = 0
                loss_sum_ctx = jnp.zeros(1)
                nb_steps_eph_ctx = 0

                for i, batch in enumerate(self.dataloader):

                    node, contexts, opt_state_ctx, loss_ctx, (nb_steps_ctx_, term1, term2, diff_ctx_) = train_step_ctx(node, contexts, contexts_old, batch, weights, opt_state_ctx, loss_key)

                    loss_sum_ctx += jnp.array([loss_ctx])
                    nb_steps_eph_ctx += nb_steps_ctx_

                    nb_batches_ctx += 1

                diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
                if diff_ctx < inner_tol_ctx or out_step==0:
                    break
                contexts_prev = contexts

            loss_epoch_ctx = loss_sum_ctx/nb_batches_ctx

            if context_callback is not None:
                context_callback(out_step + 1, contexts.params)


            losses_node.append(loss_epoch_node)
            losses_ctx.append(loss_epoch_ctx)
            nb_steps_node.append(nb_steps_eph_node)
            nb_steps_ctx.append(nb_steps_eph_ctx)

            if out_step%print_error_every==0 or out_step<=3 or out_step==nb_outer_steps_max-1:
                
                self.learner.neuralode = node
                self.learner.contexts = contexts
                if out_step > 0 and save_path:
                    self.learner.save_learner(save_path+"checkpoints/", suffix=out_step)

                if val_dataloader is not None:
                    self.learner.neuralode = node
                    self.learner.contexts = contexts
                    ind_crit, _ = tester.test(val_dataloader, int_cutoff=1.0, criterion=val_criterion, verbose=False)
                    val_losses.append(np.array([out_step, ind_crit]))
                    print(f"    Outer Step: {out_step:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}     ValIndCrit: {ind_crit:-.8f}", flush=True)
                    if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                        print(f"        Saving best model so far ...")
                        self.save_trainer(save_path)
                        self.learner.save_learner(save_path)

                else:
                    print(f"    Epoch: {out_step:-5d}      LossTrajs: {loss_epoch_node[0]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True)

                print(f"        -NbInnerStepsNode: {in_step_node+1:4d}\n        -NbInnerStepsCxt: {in_step_ctx+1:4d}\n        -InnerToleranceNode: {inner_tol_node:.2e}\n        -InnerToleranceCtx:  {inner_tol_ctx:.2e}\n        -DiffNode: {diff_node:.2e}\n        -DiffCxt:  {diff_ctx:.2e}", flush=True)

            if in_step_node < 1 and in_step_ctx < 1:
                early_stopping_count += 1
            else:
                early_stopping_count = 0

            if (patience is not None) and (early_stopping_count >= patience):
                print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
                break


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)
        print("Environment weights at the end of the training:", weights)

        self.losses_node.append(jnp.vstack(losses_node))
        self.losses_ctx.append(jnp.vstack(losses_ctx))
        self.nb_steps_node.append(jnp.array(nb_steps_node) / (in_step_node+1))
        self.nb_steps_ctx.append(jnp.array(nb_steps_ctx)/ (in_step_ctx+1))

        if val_dataloader is not None:
            self.val_losses.append(np.vstack(val_losses))

        self.opt_node_state = opt_state_node
        self.opt_ctx_state = opt_state_ctx

        self.learner.neuralode = node
        self.learner.contexts = contexts

        # Save the results and the artefacts
        if save_path:
            self.save_trainer(save_path)
            self.learner.save_learner(save_path)  # Match the final recorded contexts.

    def adapt_bulk(self, 
                    data_loader, 
                    nb_epochs, 
                    optimizer=None, 
                    print_error_every=100, 
                    save_path=False, 
                    key=None,
                    context_callback=None):
        """Adapt to new environments in bulk.

        context_callback(epoch, params) observes initialization (0) and every
        completed epoch on the host, without changing optimizer behavior.
        """

        loss_fn = self.learner.loss_fn
        node = self.learner.turn_off_self_modulation()

        if optimizer is None:       ## You may want to continue a previous adaptation !
            if hasattr(self, 'opt_adapt'):
                print("WARNING: No optimizer provided for adaptation, using any previrouly defined for adapation")
                opt = self.opt_adapt
                contexts = self.learner.contexts_adapt
                opt_state = self.opt_state_adapt
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            contexts = ContextParams(data_loader.nb_envs, self.learner.contexts.params.shape[1], key)
            opt_state = opt.init(contexts)
            self.learner.init_ctx_params_adapt = contexts.params.copy()
            self.losses_adapt = []
            self.nb_steps_adapt = []

        @eqx.filter_jit
        def train_step(node, contexts, batch, weights, opt_state, key):
            print('\nCompiling function "train_step" for context ...')

            loss_fn_ = lambda contexts, node, batch, weights, key: loss_fn(node, contexts, batch, weights, key)

            (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn_, has_aux=True)(contexts, node, batch, weights, key)

            updates, opt_state = opt.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return node, contexts, opt_state, loss, aux_data

        nb_train_steps_per_epoch = int(np.ceil(data_loader.nb_trajs_per_env / data_loader.batch_size))
        total_steps = nb_epochs * nb_train_steps_per_epoch

        print(f"\n\n=== Beginning adaptation ... ===")
        print(f"    Number of examples in a batch: {data_loader.batch_size}")
        print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
        print(f"    Number of training epochs: {nb_epochs}")
        print(f"    Total number of training steps: {total_steps}")

        start_time = time.time()

        losses = []
        nb_steps = []

        weights = jnp.ones(data_loader.nb_envs) / data_loader.nb_envs
        loss_key = get_new_key(key)

        if context_callback is not None:
            context_callback(0, contexts.params)

        for epoch in range(nb_epochs):
            nb_batches = 0
            loss_sum = jnp.zeros(1)
            nb_steps_eph = 0

            for i, batch in enumerate(data_loader):
                loss_key = get_new_key(loss_key)

                node, contexts, opt_state, loss, (nb_steps_, term1, term2) = train_step(node, contexts, batch, weights, opt_state, loss_key)

                loss_sum += jnp.array([loss])
                nb_steps_eph += nb_steps_

                nb_batches += 1

            loss_epoch = loss_sum/nb_batches

            if context_callback is not None:
                context_callback(epoch + 1, contexts.params)

            losses.append(loss_epoch)
            nb_steps.append(nb_steps_eph)

            if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:
                print(f"    Epoch: {epoch:-5d}     LossContext: {loss_epoch[0]:-.8f}", flush=True)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_adapt.append(jnp.vstack(losses))
        self.nb_steps_adapt.append(jnp.array(nb_steps))

        self.opt_adapt = opt
        self.opt_state_adapt = opt_state

        self.learner.contexts_adapt = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

    def adapt_sequential(self, 
                         data_loader, 
                         nb_epochs, 
                         optimizer=None, 
                         print_error_every=100, 
                         save_path=False, 
                         key=None,
                         context_callback=None):
        """Adapt to new environments sequentially.

        context_callback(epoch, params, env_id=env_id) observes initialization
        and completed local epochs for each environment in dataset order.
        """

        loss_fn = self.learner.loss_fn
        node = self.learner.turn_off_self_modulation()

        if optimizer is None:       ## You want to continue a previous adaptation !!!
            if hasattr(self, 'opt_adapt'):
                print("WARNING: No optimizer provided for adaptation, using any previrouly defined for adapation")
                opt = self.opt_adapt
                contexts = self.learner.contexts_adapt
                opt_state = self.opt_state_adapt
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            self.losses_adapt = []
            self.nb_steps_adapt = []

        @eqx.filter_jit
        def train_step(node, contexts, batch, weights, opt_state, key):
            print('\nCompiling function "train_step" for context ...')

            loss_fn_ = lambda contexts, node, batch, weights, key: loss_fn(node, contexts, batch, weights, key)

            (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn_, has_aux=True)(contexts, node, batch, weights, key)

            updates, opt_state = opt.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return node, contexts, opt_state, loss, aux_data

        nb_train_steps_per_epoch = int(np.ceil(data_loader.nb_trajs_per_env / data_loader.batch_size))
        total_steps = nb_epochs * nb_train_steps_per_epoch

        print(f"\n\n=== Beginning sequential adaptation ... ===")
        print(f"    Number of examples in a batch: {data_loader.batch_size}")
        print(f"    Number of train steps per epoch: {nb_train_steps_per_epoch}")
        print(f"    Number of training epochs: {nb_epochs}")
        print(f"    Total number of training steps: {total_steps}")

        nb_adapt_envs = data_loader.nb_envs

        contexts = []
        all_losses = []
        all_nb_steps = []
        inits_ctx = []
        for env_id in range(nb_adapt_envs):

            start_time = time.time()

            print(f"\nAdapting to environment {env_id} ...")

            new_dataset = data_loader.dataset[env_id:env_id+1,...]

            new_dataloader = DataLoader(new_dataset, t_eval=data_loader.t_eval, adaptation=True, key=key)

            opt = optimizer
            context = ContextParams(1, self.learner.contexts.params.shape[1], key)
            inits_ctx.append(context.params)
            opt_state = opt.init(context)

            losses = []
            nb_steps = []

            weights = jnp.ones(1)
            loss_key = get_new_key(key)

            if context_callback is not None:
                context_callback(0, context.params, env_id=env_id)

            for epoch in range(nb_epochs):
                nb_batches = 0
                loss_sum = jnp.zeros(1)
                nb_steps_eph = 0

                for i, batch in enumerate(new_dataloader):
                    loss_key = get_new_key(loss_key)

                    node, context, opt_state, loss, (nb_steps_, term1, term2) = train_step(node, context, batch, weights, opt_state, loss_key)

                    loss_sum += jnp.array([loss])
                    nb_steps_eph += nb_steps_

                    nb_batches += 1

                loss_epoch = loss_sum/nb_batches

                if context_callback is not None:
                    context_callback(epoch + 1, context.params, env_id=env_id)

                losses.append(loss_epoch)
                nb_steps.append(nb_steps_eph)

                if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:
                    print(f"    Epoch: {epoch:-5d}     LossContext: {loss_epoch[0]:-.8f}", flush=True)

            wall_time = time.time() - start_time
            time_in_hmsecs = seconds_to_hours(wall_time)
            print("\nGradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

            contexts.append(context.params)
            all_losses.append(jnp.stack(losses))
            all_nb_steps.append(jnp.array(nb_steps))


        contexts = eqx.tree_at(lambda c: c.params, 
                                ContextParams(nb_adapt_envs, self.learner.contexts.params.shape[1], None), 
                                jnp.concatenate(contexts))

        self.losses_adapt.append(jnp.mean(jnp.stack(all_losses), axis=0))
        self.nb_steps_adapt.append(jnp.mean(jnp.stack(all_nb_steps), axis=0))

        self.opt_adapt = opt
        self.opt_state_adapt = opt.init(contexts)

        self.learner.contexts_adapt = contexts
        self.learner.init_ctx_params_adapt = jnp.concatenate(inits_ctx)

        if save_path:
            self.save_adapted_trainer(save_path)

#%%
### ORIGINAL MODEL AND LOSS ###
class NeuralNet(eqx.Module):
    """ Nueral Network for the neural ODE's vector field """
    layers_data: list
    layers_context: list
    layers_shared: list
    activations: list

    def __init__(self, data_size, int_size, context_size, key=None):
        keys = generate_new_keys(key, num=12)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        self.layers_context = [eqx.nn.Linear(context_size, context_size//4, key=keys[0]), self.activations[0],
                               eqx.nn.Linear(context_size//4, int_size, key=keys[1]), self.activations[1], eqx.nn.Linear(int_size, int_size, key=keys[2])]

        self.layers_data = [eqx.nn.Linear(data_size, int_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(int_size, int_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(int_size, int_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(2*int_size, int_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(int_size, int_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(int_size, int_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(int_size, data_size, key=keys[9])]

    def __call__(self, t, y, ctx):

        for layer in self.layers_context:
            ctx = layer(ctx)

        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        return y

## Define a loss function for one environment (one context)
def loss_fn_env(model, trajs, t_eval, ctx, all_ctx_s, key):

    ## Define the context pool using the Random-All strategy
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction loss
    term2 = jnp.mean(jnp.abs(ctx))              ## context regularisation
    term3 = params_norm_squared(model)          ## weight regularisation

    loss_val = term1 + 1e-3*term2 + 1e-3*term3

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)


#%%
### SELECT OR RESTORE THE EXPERIMENT ###
runs_folder.mkdir(exist_ok=True)
if run_folder is None:
    if train:
        from datetime import datetime
        run_path = runs_folder / datetime.now().strftime("%y%m%d-%H%M%S-%f-ctx")
    elif script_path.parent.parent == runs_folder:
        run_path = script_path.parent
    else:
        completed_runs = sorted(runs_folder.glob("*/context_animation/completed.json"),
                                key=lambda p: p.stat().st_mtime)
        if not completed_runs:
            raise FileNotFoundError("No completed context run found. Set run_folder to a saved run in runs/.")
        run_path = completed_runs[-1].parent.parent
else:
    run_path = Path(run_folder).expanduser()
    if not run_path.is_absolute():
        run_path = example_folder / run_path
run_path = run_path.resolve()
if runs_folder.resolve() not in run_path.parents:
    raise ValueError("run_folder must be a subfolder of this example's runs/ directory")
run_folder = str(run_path) + "/"
adapt_folder = str(run_path / "adapt") + "/"
context_folder = run_path / "context_animation"

# Save model/training settings so train=False restores the matching experiment.
# Animation settings stay editable above and are not overwritten on reload.
setting_names = (
    "seed", "context_pool_size", "context_size", "ncf_variant",
    "nb_outer_steps_max", "nb_inner_steps_max", "proximal_beta",
    "inner_tol_node", "inner_tol_ctx", "early_stopping_patience",
    "init_lr", "sched_factor", "ivp_args", "sequential_adapt",
    "init_lr_adapt", "sched_factor_adapt", "nb_epochs_adapt", "data_folder",
)
if train:
    run_path.mkdir(parents=True, exist_ok=False)
    for name in ("adapt", "checkpoints", "context_animation"):
        (run_path / name).mkdir()
    shutil.copy2(script_path, run_path / script_path.name)
    settings = {name: globals()[name] for name in setting_names}
    (run_path / "settings.json").write_text(json.dumps(settings, indent=2))
    np.savez(context_folder / "environment_parameters.npz", train=TRAIN_PARAMETERS,
             adapt=ADAPT_PARAMETERS, names=np.array(["beta", "delta"]))
else:
    settings_path = run_path / "settings.json"
    if not settings_path.is_file():
        raise FileNotFoundError(f"No single-file experiment settings found in {run_path}")
    settings = json.loads(settings_path.read_text())
    for name in setting_names:
        globals()[name] = settings[name]
    # Validate before restoring; never silently plot an interrupted trajectory.
    load_history(context_folder / "train_contexts.npy")
    if adapt_test and adapt_restore:
        load_history(context_folder / "adapt_contexts.npy")
    print("Restoring saved experiment settings from:", settings_path)
taylor_order = ncf_variant
print(f"NCF variant: NCF-t{ncf_variant}; proximal optimisation")
print("Run folder:", run_folder)

#%%
## Generate missing data using the unchanged dataset.py.
for split, filename in (("train", "train.npz"), ("test", "test.npz")):
    if not (Path(data_folder) / filename).exists():
        subprocess.run([sys.executable, str(example_folder / "dataset.py"),
                        "--split=" + split, "--savepath=" + data_folder], check=True)
train_dataloader = DataLoader(data_folder + "train.npz", shuffle=True, key=seed)
val_dataloader = DataLoader(data_folder + "test.npz", shuffle=False)
nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size
if nb_envs != len(TRAIN_PARAMETERS):
    raise ValueError("Expected dataset.py's nine training environments, in their original order")
print("Number of environments:", nb_envs)
print("Number of trajectories per environment:", nb_trajs_per_env)
print("Number of steps per trajectory:", nb_steps_per_traj)
print("Data size:", data_size)

## Create the same neural network, vector field, contexts, and learner as main.py.
neuralnet = NeuralNet(data_size=2, int_size=64, context_size=context_size, key=seed)
vectorfield = SelfModulatedVectorField(physics=None, augmentation=neuralnet, taylor_order=taylor_order)
contexts = ContextParams(nb_envs, context_size, key=None)
learner = Learner(vectorfield, contexts, loss_fn_env, integrator, ivp_args, key=seed)
nb_total_epochs = nb_outer_steps_max * nb_inner_steps_max
sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
    boundaries_and_scales={nb_total_epochs//3: sched_factor, 2*nb_total_epochs//3: sched_factor})
sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr,
    boundaries_and_scales={nb_total_epochs//3: sched_factor, 2*nb_total_epochs//3: sched_factor})
opt_node, opt_ctx = optax.adam(sched_node), optax.adam(sched_ctx)
trainer = ContextRecordingTrainer(train_dataloader, learner, (opt_node, opt_ctx), key=seed)

#%%
## NCF-t1 uses proximal alternating minimisation in this experiment.
if train:
    train_history = ContextHistory(context_folder / "train_contexts.npy",
                                   nb_outer_steps_max, nb_envs, context_size)
    trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max,
                          nb_inner_steps_max=nb_inner_steps_max,
                          proximal_reg=proximal_beta,
                          inner_tol_node=inner_tol_node,
                          inner_tol_ctx=inner_tol_ctx,
                          print_error_every=print_error_every,
                          save_path=run_folder if save_trainer else False,
                          val_dataloader=val_dataloader,
                          patience=early_stopping_patience,
                          int_prop=1.0, key=seed,
                          context_callback=train_history)
else:
    trainer.restore_trainer(path=run_folder)

#%%
## Evaluate the model on the original in-domain test set.
visualtester = VisualTester(trainer)
ind_crit, _ = visualtester.test(val_dataloader, int_cutoff=1.0)
visualtester.visualize(val_dataloader, int_cutoff=1.0,
                       save_path=run_folder + "results_in_domain.png")

#%%
## Adapt on ood_train.npz, evaluate on ood_test.npz, and record each local epoch.
if adapt_test:
    for split, filename in (("adapt", "ood_train.npz"), ("adapt_test", "ood_test.npz")):
        if not (Path(data_folder) / filename).exists():
            subprocess.run([sys.executable, str(example_folder / "dataset.py"),
                            "--split=" + split, "--savepath=" + data_folder], check=True)
    adapt_dataloader = DataLoader(data_folder + "ood_train.npz", adaptation=True, key=seed)
    adapt_dataloader_test = DataLoader(data_folder + "ood_test.npz", adaptation=True, key=seed)
    if adapt_dataloader.nb_envs != len(ADAPT_PARAMETERS):
        raise ValueError("Expected dataset.py's four adaptation environments in their original order")
    if adapt_restore:
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)
    else:
        sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr_adapt,
            boundaries_and_scales={nb_epochs_adapt//3: sched_factor_adapt,
                                   2*nb_epochs_adapt//3: sched_factor_adapt})
        opt_adapt = optax.adabelief(sched_ctx_new)
        adapt_history = ContextHistory(context_folder / "adapt_contexts.npy",
                                       nb_epochs_adapt, adapt_dataloader.nb_envs, context_size)
        adapt_method = trainer.adapt_sequential if sequential_adapt else trainer.adapt_bulk
        adapt_method(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt,
                     print_error_every=print_error_every, save_path=adapt_folder,
                     key=seed, context_callback=adapt_history)
    ood_crit, _ = visualtester.test(adapt_dataloader_test, int_cutoff=1.0)
    visualtester.visualize(adapt_dataloader_test, int_cutoff=1.0,
                           save_path=adapt_folder + "results_ood.png")

#%%
## The probe uses only final training contexts and their nine (beta, delta) labels.
## PCA is fixed across epochs; t-SNE is one joint embedding of sampled snapshots.
## Sequential adaptation movies align environments by their local epoch.
if animate_contexts:
    visualise_contexts(context_folder,
                       load_history(context_folder / "train_contexts.npy"),
                       np.asarray(trainer.learner.contexts.params),
                       load_history(context_folder / "adapt_contexts.npy") if adapt_test else None,
                       max_frames=animation_max_frames, seed=seed, sequential=sequential_adapt)
if train:
    (context_folder / "completed.json").write_text(json.dumps({
        "outer_epochs": nb_outer_steps_max, "adapt_epochs": nb_epochs_adapt if adapt_test else 0,
        "model_saved": save_trainer,
    }, indent=2))
print("Results saved in:", run_folder)
