from ._utils import *


class VisualTester:
    def __init__(self, trainer, key=None):
        self.key = get_new_key(key)
        self.trainer = trainer







    def test(self, data_loader, criterion=None, int_cutoff=1.0, verbose=True):
        """ Compute test metrics on the adaptation dataloader  """

        criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[:, :, :test_length, :]
        t_test = t_eval[:test_length]

        if verbose == True:
            if data_loader.adaptation == False:
                print("==  Begining in-domain testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
            else:
                print("==  Begining out-of-distribution testing ... ==")
                print("    Number of training environments:", self.trainer.dataloader.nb_envs)
                print("    Number of adaptation environments:", data_loader.nb_envs)
            print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
            print("    Length of the testing trajectories:", test_length)

        if data_loader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params

        neuralode = self.trainer.learner.turn_off_self_modulation()
        X_hat, _ = jax.vmap(neuralode, in_axes=(0, None, 0, 0))(X[:, :, 0, :], 
                                                                t_test, 
                                                                contexts,
                                                                contexts)

        batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

        crit_all = batched_criterion(X, X_hat).mean(axis=1)
        crit = crit_all.mean(axis=0)

        if verbose == True:
            if data_loader.adaptation == False:
                print("Test Score (In-Domain):", crit)
            else:
                print("Test Score (OOD):", crit)
            print(flush=True)

        return crit, crit_all







    def visualize(self, 
                  data_loader, 
                  e=None, 
                  traj=None, 
                  dims=(0,1), 
                  context_dims=(0,1), 
                  int_cutoff=1.0, 
                  save_path=False, 
                  key=None):

        e_key, traj_key = get_new_key(time.time_ns(), num=2)
        e = e if e is not None else jax.random.randint(e_key, (1,), 0, data_loader.nb_envs)[0]
        traj = traj is not None if traj else jax.random.randint(traj_key, (1,), 0, data_loader.nb_trajs_per_env)[0]

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[e, traj:traj+1, :test_length, :]
        t_test = t_eval[:test_length]

        if data_loader.adaptation == False:
            print("==  Begining in-domain visualisation ... ==")
        else:
            print("==  Begining out-of-distribution visualisation ... ==")
        print("    Environment id:", e)
        print("    Trajectory id:", traj)
        print("    Visualized dimensions:", dims)
        print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
        print("    Length of the testing trajectories:", test_length)

        if data_loader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params

        model = self.trainer.learner.turn_off_self_modulation()
        X_hat, _ = model(X[:, 0, :], t_test, contexts[e], contexts[e])

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

        mks = 2
        dim0, dim1 = dims

        ax['A'].plot(t_test, X[:, 0], c="deepskyblue", label=f"$x_{{{dim0}}}$ (GT)")
        ax['A'].plot(t_test, X_hat[:, 0], "o", c="royalblue", label=f"$\\hat{{x}}_{{{dim0}}}$ (NCF)", markersize=mks)

        ax['A'].plot(t_test, X[:, 1], c="violet", label=f"$x_{{{dim1}}}$ (GT)")
        ax['A'].plot(t_test, X_hat[:, 1], "x", c="purple", label=f"$\\hat{{x}}_{{{dim1}}}$ (NCF)", markersize=mks)

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
        ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="NCF")
        ax['B'].set_xlabel(f"$x_{{{dim0}}}$")
        ax['B'].set_ylabel(f"$x_{{{dim1}}}$")
        ax['B'].set_title("Phase space")
        ax['B'].legend()

        nb_envs = data_loader.nb_envs

        nb_steps = np.concatenate(self.trainer.nb_steps_node)
        losses_node = np.vstack(self.trainer.losses_node)
        losses_ctx = np.vstack(self.trainer.losses_ctx)
        xis = self.trainer.learner.contexts.params
        init_xis = self.trainer.learner.init_ctx_params

        if data_loader.adaptation == True:  ## Overwrite the above if adaptation
            nb_steps = np.concatenate(self.trainer.nb_steps_adapt)
            losses_node = np.vstack(self.trainer.losses_adapt)      ## Replotting the label context !
            losses_ctx = np.vstack(self.trainer.losses_adapt)
            xis = self.trainer.learner.contexts_adapt.params
            init_xis = self.trainer.learner.init_ctx_params_adapt

        mke = np.ceil(losses_node.shape[0]/100).astype(int)

        label_node = "Node Loss" if data_loader.adaptation == False else "Node Loss Adapt"
        ax['C'].plot(losses_node[:,0], label=label_node, color="grey", linewidth=3, alpha=1.0)
        label_ctx = "Context Loss" if data_loader.adaptation == False else "Context Loss Adapt"
        ax['C'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)

        if data_loader.adaptation==False and hasattr(self.trainer, 'val_losses') and len(self.trainer.val_losses)>0:
            val_losses = np.vstack(self.trainer.val_losses)
            ax['C'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['C'].set_xlabel("Epochs")
        ax['C'].set_title("Loss Terms")
        ax['C'].set_yscale('log')
        ax['C'].legend()

        ax['D'].plot(nb_steps, c="brown")
        ax['D'].set_xlabel("Epochs")
        ax['D'].set_title("Total Number of Steps Taken (Proportional to NFEs)")
        if np.all(nb_steps>0):
            ax['D'].set_yscale('log')

        eps = 0.1
        colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
        colors = colors*(nb_envs)
        cdim0, cdim1 = context_dims

        ax['E'].scatter(init_xis[:,cdim0], init_xis[:,cdim1], s=30, c=colors[:nb_envs], marker='X')
        ax['F'].scatter(xis[:,cdim0], xis[:,cdim1], s=50, c=colors[:nb_envs], marker='o')
        for i, (x, y) in enumerate(init_xis[:, context_dims]):
            ax['E'].annotate(str(i), (x, y), fontsize=8)
        for i, (x, y) in enumerate(xis[:, context_dims]):
            ax['F'].annotate(str(i), (x, y), fontsize=8)
        ax['E'].set_title(r'Initial Contexts')
        ax['E'].set_xlabel(f'dim {cdim0}')
        ax['E'].set_ylabel(f'dim {cdim1}')

        ax['F'].set_title(r'Final Contexts')
        ax['F'].set_xlabel(f'dim {cdim0}')
        ax['F'].set_ylabel(f'dim {cdim1}')

        plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);








    def visualizeUQ(self, 
                    data_loader, 
                    e=None, 
                    traj=None, 
                    dims=(0,1), 
                    std_scale=1e2, 
                    int_cutoff=1.0, 
                    save_path=False, 
                    forecast=True, 
                    verbose=True, 
                    title=True, 
                    key=None):
        """ Visualise the results of the neural ODE model with epistemic uncertainty quantification """

        # assert data_loader.nb_envs == self.trainer.dataloader.nb_envs, "The number of environments in the test dataloader must be the same as the number of environments in the trainer."

        e_key, traj_key = get_new_key(time.time_ns(), num=2)
        e = e if e is not None else jax.random.randint(e_key, (1,), 0, data_loader.nb_envs)[0]
        traj = traj is not None if traj else jax.random.randint(traj_key, (1,), 0, data_loader.nb_trajs_per_env)[0]

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[e, traj:traj+1, :test_length, :]
        t_test = t_eval[:test_length]

        if forecast == True:
            delta_t = t_eval[1] - t_eval[0]
            t_span_ext = (t_eval[0], t_eval[-1]+delta_t+ (t_eval[-1]+delta_t-t_eval[0])/2)
            # t_test_ext = jnp.linspace(*t_span_ext, 2*test_length)
            t_test_ext = jnp.linspace(t_eval[-1], t_span_ext[-1], 2+test_length//2, endpoint=True)
            t_test_ext = jnp.concatenate([t_test, t_test_ext[1:]])
            ## Round to two decimal places
            t_test_ext = jnp.round(t_test_ext, 3)

        else:
            t_test_ext = t_test

        if verbose == True:
            if data_loader.adaptation == False:
                print("==  Begining in-domain visualisation with UQ... ==")
            else:
                print("==  Begining out-of-distribution visualisation with UQ ... ==")
            print("    Environment id:", e)
            print("    Trajectory id:", traj)
            print("    Visualized dimensions:", dims)
            print("    Final length of the training trajectories:", self.trainer.dataloader.int_cutoff)
            print("    Length of the testing trajectories:", test_length)

        contexts = self.trainer.learner.contexts.params
        if data_loader.adaptation == True:
            contexts_adapt = self.trainer.learner.contexts_adapt.params
            print("Shapes before concatenation:", contexts_adapt.shape, contexts.shape)
            contexts = jnp.concatenate([contexts_adapt, contexts], axis=0)

        model = self.trainer.learner.turn_off_self_modulation()
        batched_neuralode = jax.vmap(model, in_axes=(None, None, None, 0))

        X_hat, _ = batched_neuralode(X[:, 0, :], 
                                     t_test_ext, 
                                     contexts[e], 
                                     contexts)

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        fig, ax = plt.subplot_mosaic('A;B', figsize=(6*1, 4*2))

        mks = 2
        dim0, dim1 = dims

        ## Plot in axis C and D. Same as above, but the mean and std across X_hat's first dimension
        X_hat_mean = X_hat.mean(axis=0)
        X_hat_std = std_scale*X_hat.std(axis=0)

        ax['A'].plot(t_test, X[:, dim0], "o", c="royalblue", label=f"$x_{{{dim0}}}$ (GT)")
        ax['A'].plot(t_test_ext, X_hat_mean[:, dim0], c="deepskyblue", label=f"$\\hat{{x}}_{{{dim0}}}$ (NCF)", markersize=mks)
        ax['A'].fill_between(t_test_ext, X_hat_mean[:, dim0]-X_hat_std[:, dim0], X_hat_mean[:, dim0]+X_hat_std[:, dim0], color="deepskyblue", alpha=0.2)

        ax['A'].plot(t_test, X[:, dim1], "x", c="purple", label=f"$x_{{{dim1}}}$ (GT)")
        ax['A'].plot(t_test_ext, X_hat_mean[:, dim1], c="violet", label=f"$\\hat{{x}}_{{{dim1}}}$ (NCF)", markersize=mks)
        ax['A'].fill_between(t_test_ext, X_hat_mean[:, dim1]-X_hat_std[:, dim1], X_hat_mean[:, dim1]+X_hat_std[:, dim1], color="violet", alpha=0.2)

        ## If forecasting, place a vertical line to indicate when the forecast starts
        if forecast == True:
            ax['A'].axvline(x=t_eval[-1], color='crimson', linestyle='--', label="Forecast Start")

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, dim0], X[:, dim1], ".", c="teal", label="GT")
        ax['B'].plot(X_hat_mean[:, dim0], X_hat_mean[:, dim1], c="turquoise", label="NCF")
        ax['B'].fill_between(X_hat_mean[:, dim0], X_hat_mean[:, dim1]-X_hat_std[:, dim1], X_hat_mean[:, dim1]+X_hat_std[:, dim1], color="turquoise", alpha=0.2)

        ax['B'].set_xlabel(f"$x_{{{dim0}}}$")
        ax['B'].set_ylabel(f"$x_{{{dim1}}}$")
        ax['B'].set_title("Phase space")
        ax['B'].legend()

        if title:
            plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);








    def printUQ_metrics(self, 
                        data_loader, 
                        forecast_factor=0.5, 
                        conf_level_scale=3, 
                        nb_bins=12, 
                        std_color=None, 
                        max_dot_size=None, 
                        save_path=False):

        """ 
        Calculate a few UQ metrics the results of the neural ODE model with epistemic uncertainty quantification 
        """

        print("==  Begining in-domain visualisation with UQ... ==")

        X = data_loader.dataset
        t_eval = data_loader.t_eval
        test_length = t_eval.shape[0]
        t_test = t_eval

        delta_t = t_eval[1] - t_eval[0]
        t_span_ext = (t_eval[0], t_eval[-1]+delta_t+ forecast_factor*(t_eval[-1]+delta_t-t_eval[0]))
        t_test_ext = jnp.linspace(t_eval[-1], t_span_ext[-1], 2+test_length//2, endpoint=False)
        t_test_ext = jnp.concatenate([t_test, t_test_ext[1:]])
        t_test_ext = jnp.round(t_test_ext, 3)

        contexts_ind = self.trainer.learner.contexts.params
        if data_loader.adaptation == True:
            contexts_ood = self.trainer.learner.contexts_adapt.params
            contexts_all = jnp.concatenate([contexts_ind, contexts_ood], axis=0)
            contexts = contexts_ood     ## Context of interest
            std_color = "crimson" if std_color is None else std_color
        else:
            contexts_all = contexts_ind
            contexts = contexts_ind
            std_color = "royalblue" if std_color is None else std_color

        model = self.trainer.learner.turn_off_self_modulation()
        @eqx.filter_vmap
        def UQ_metrics(X_e, contexts_e):
            ## "==  Uncertainty Quantification Metrics for many traj in one env =="
            ## The first 3 require the gound truth, the final one doesnt, we plot that one
            ## X_e: (trajs, time, dim)
            ## context_e: (dim,)
  
            batched_neuralode = jax.vmap(model, in_axes=(None, None, None, 0))
            X_hat_ext, _ = batched_neuralode(X_e[:, 0, :], t_test_ext, contexts_e, contexts_all)    ## X_hat_ext: (envs, trajs, time_ext, dim)
            X_hat = X_hat_ext[:, :, :test_length, :]

            means = jnp.mean(X_hat, axis=0, keepdims=False)
            std = jnp.std(X_hat, axis=0, keepdims=False, ddof=0)

            # 1. Relative MSE loss: Difference between the mean of the predictions and the ground truth
            # rel_mse_loss = jnp.mean(jnp.mean((X_e-means)**2, axis=(1,2)) / jnp.mean(means**2, axis=(1,2)))
            # rel_mse_loss = jnp.mean((X_e-means)**2 / (X_e**2))
            # rel_mse_loss = jnp.mean((X_e-means)**2 / (X_e**2+1e-6))

            ## Design a denominator that is never zero
            denom = jnp.where(jnp.abs(X_e) > 1e-3, jnp.abs(X_e), jnp.inf)
            rel_mse_loss = jnp.mean((X_e-means)**2 / denom**2)

            # 2. Relative MAPE: same as abopve but in percentage
            rel_mape_loss = jnp.mean(jnp.abs(X_e-means)/(denom))

            # 3. Confidence level: the percentage of the predictions that fall within the 3xstd of predictions
            conf_level = jnp.mean(jnp.mean(jnp.abs(X_e-means) <= conf_level_scale*std, axis=(1,2)))

            # 4. Relative standard deviation: quotient the std of the predictions to the mean of the predictions
            long_std = jnp.std(X_hat_ext, axis=0)
            rel_std_loss = jnp.mean(jnp.mean(long_std**2, axis=-1), axis=0)

            return rel_mse_loss, rel_mape_loss, conf_level, rel_std_loss, (X_e-means, std)

        rel_mse_loss, rel_mape_loss, conf_level, rel_std_loss, aux_dat = UQ_metrics(X, contexts)
        m_rel_mse, m_rel_mape, m_conf_level = jnp.mean(rel_mse_loss), jnp.mean(rel_mape_loss), jnp.mean(conf_level)
        m_rel_std = jnp.mean(rel_std_loss, axis=0)

        ## Print the results properly
        print("==  Uncertainty Quantification Metrics (across all environments)  ==")
        print(f"    Relative MSE Loss: {m_rel_mse*100:.3f} %")
        print(f"    MAPE Loss:         {m_rel_mape*100:.2f} %")
        print(f"    Confidence Level:  {m_conf_level*100:.2f} % - (also called the empirical coverage probability)")

        ## Plot the latest calculation
        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 4*3))
        for e in range(rel_std_loss.shape[0]):
            ax.plot(t_test_ext, rel_std_loss[e], "s-", label=f"Env {e+1}")
        ax.axvline(x=t_eval[-1], color='crimson', linestyle='--', label="Forecast Start")

        # ax.set_title("Relative Standard Deviation")
        ax.set_xlabel(f"Time $t$")
        ax.set_ylabel(r'$\sum \Vert\sigma\Vert_2$')
        ax.legend()

        ## Plot scatter the auxiliary data
        errors, deviations = aux_dat
        errors, deviations = jnp.abs(errors.flatten()), deviations.flatten()
        ax2.scatter(deviations, errors, s=1, color=std_color)
        # ax2.set_title("Errors vs Deviations")
        ax2.set_xlabel(f"$\hat \sigma$")
        ax2.set_ylabel(f"$| x - \hat \mu |$")

        ## Put the tandard deviations in bins, and plot the mean error in each bin (plot the std as well as a vertical bar)
        bins = np.linspace(deviations.min(), deviations.max(), nb_bins)
        digitized = np.digitize(deviations, bins)
        bin_means = [errors[digitized == i].mean() for i in range(1, len(bins))]
        bin_stds = [errors[digitized == i].std()/2 for i in range(1, len(bins))]
        bin_counts_raw = [len(errors[digitized == i])//5 for i in range(1, len(bins))]
        ## Normalise the bin counts between 1 and the max
        max_size = max_dot_size if max_dot_size else max(bin_counts_raw)
        bin_counts = np.interp(bin_counts_raw, (min(bin_counts_raw), max(bin_counts_raw)), (10, max_size))

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax3.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='s', markersize=0, color=std_color)
        ## Draw the dots using scatter
        ax3.scatter(bin_centers, bin_means, s=bin_counts, alpha=0.5, color=std_color)
        ## Set the xticks and their labels at the centers
        ax3.set_xticks(bin_centers)
        ax3.set_xticklabels([f"{b:.3f}" for b in bin_centers])

        ## Plot rectangles that span the standard deviation bins
        for i in range(len(bins)-1):
            ax3.add_patch(patches.Rectangle((bins[i], 0), bins[i+1]-bins[i], bin_means[i], color='grey', alpha=0.1))

        # ax3.set_title("Errors vs Deviations")
        ax3.set_ylabel(f"Absolute Error $ | x - \hat \mu |$")
        ax3.set_xlabel(f"Standard Deviation $\hat \sigma$")

        ## Print and/or return those means (we will plot the rel_std_loss trajectories InD and OoD for all 6 problems)
        if save_path:
            np.savez(save_path+".npz", rel_mse=m_rel_mse, rel_mape=m_rel_mape, conf_level=m_conf_level, rel_std=m_rel_std, deviations=deviations, errors=errors)
            fig.savefig(save_path+".svg", dpi=100, bbox_inches='tight')
            print("Matrics plots saved in:", save_path);

        return m_rel_mse, m_rel_mape, m_conf_level, m_rel_std








    def visualize2D(self, 
                    data_loader, 
                    e=None, 
                    traj=None, 
                    res=(32,32), 
                    int_cutoff=1.0, 
                    nb_plot_timesteps=10, 
                    cmap='gist_ncar', 
                    save_path=False, 
                    key=None):

        """
        The visualize2D function is used to visualize the results of a trained NCF on spatio-temporal data like PDEs (2D).
        
        :param data_loader: data on which to predict and visualize the results
        :param e: The environment to visualize
        :param traj: Specify which trajectory to visualize
        :param res: Specify the resolution of the 2D grid
        :param int_cutoff: Specify the length of the trajectory to be visualized
        :param nb_plot_timesteps: Specify the number of timesteps to be visualized
        :param save_path: Specify where to save the figure
        :param key: A random key/seed for jax
        :return: A figure with two subplots
        """
        seed = key if key else time.time_ns()%2**32
        e_key, traj_key = get_new_key(seed, num=2)
        e = e if e else jax.random.randint(e_key, (1,), 0, data_loader.nb_envs)[0]
        traj = traj if traj else jax.random.randint(traj_key, (1,), 0, data_loader.nb_trajs_per_env)[0]

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[e, traj:traj+1, :test_length, :]
        t_test = t_eval[:test_length]

        if data_loader.adaptation == False:
            print("==  Begining in-domain 2D visualisation ... ==")
        else:
            print("==  Begining out-of-distribution 2D visualisation ... ==")
        print("    Environment id:", e)
        print("    Trajectory id:", traj)
        print("    Length of the testing trajectories:", test_length)

        if data_loader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params
        model = self.trainer.learner.turn_off_self_modulation()
        X_hat, _ = model(X[:, 0, :], t_test, contexts[e], contexts[e])

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        nb_mats = X_hat.shape[1] // (res*res)
        assert nb_mats > 0, f"Not enough dimensions to form a {res}x{res} matrix"

        if test_length < nb_plot_timesteps:
            print(f"Warning: trajectory visualisation length={test_length} is less than number of plots per row={nb_plot_timesteps}.")
            nb_plot_timesteps = 1
            print(f"Setting the number of plots per row to {nb_plot_timesteps}")
        elif test_length%nb_plot_timesteps !=0:
            print(f"Warning: trajectory visualisation length={test_length} is not divisible by number of plots per row={nb_plot_timesteps}.")
            nb_plot_timesteps = int(test_length / (test_length//nb_plot_timesteps))
            print(f"Setting the number of plots per row to {nb_plot_timesteps}")

        fig, ax = plt.subplots(nrows=nb_mats*2, ncols=nb_plot_timesteps, figsize=(2*nb_plot_timesteps, 2*nb_mats*2))
        for j in range(0, test_length, test_length//nb_plot_timesteps):
            gt_j = vec_to_mats(X[j], res, nb_mats)
            ncf_j = vec_to_mats(X_hat[j], res, nb_mats)
            for i in range(nb_mats):
                ax[2*i, j].imshow(gt_j[i], cmap=cmap, interpolation='bilinear', origin='lower')
                ax[2*i+1, j].imshow(ncf_j[i], cmap=cmap, interpolation='bilinear', origin='lower')

        ## Remove the ticks and labels
        for a in ax.flatten():
            a.set_xticks([])
            a.set_yticks([])
            a.set_xticklabels([])
            a.set_yticklabels([])

        plt.suptitle(f"2D visualisation results for env={e}, traj={traj}", fontsize=20)

        plt.tight_layout()
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            np.savez(save_path+"_data.npz", X=X, X_hat=X_hat)
            print("Testing finished. Figure saved in:", save_path);
