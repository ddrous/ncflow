from ._utils import *


class VisualTester:
    def __init__(self, trainer, key=None):
        self.key = get_new_key(key)

        # self.dataloader = test_dataloader
        self.trainer = trainer

        # assert self.dataloader.nb_envs == self.trainer.dataloader.nb_envs, "The number of environments in the test dataloader must be the same as the number of environments in the trainer."

    # def test_node(self, criterion=None, int_cutoff=1.0):
    #     """ Compute test metrics on entire test dataloader  """

    #     criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

    #     t_eval = self.dataloader.t_eval
    #     test_length = int(self.dataloader.nb_steps_per_traj*int_cutoff)
    #     X = self.dataloader.dataset[:, :, :test_length, :]
    #     t_test = t_eval[:test_length]

    #     print("==  Begining testing ... ==")
    #     print("    Length of the training trajectories:", self.trainer.dataloader.int_cutoff)
    #     print("    Length of the testing trajectories:", test_length)

    #     X_hat, _ = jax.vmap(self.trainer.learner.neuralode, in_axes=(0, None, 0))(X[:, :, 0, :], 
    #                                          t_test, 
    #                                          self.trainer.learner.contexts.params)

    #     batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

    #     return batched_criterion(X_hat, X).mean(axis=1).sum(axis=0)



    # def test_cf(self, criterion=None, int_cutoff=1.0):
    #     """ Compute test metrics on entire test dataloader  """

    #     criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

    #     t_eval = self.dataloader.t_eval
    #     test_length = int(self.dataloader.nb_steps_per_traj*int_cutoff)
    #     X = self.dataloader.dataset[:, :, :test_length, :]
    #     t_test = t_eval[:test_length]

    #     print("==  Begining testing ... ==")
    #     print("    Length of the training trajectories:", self.trainer.dataloader.int_cutoff)
    #     print("    Length of the testing trajectories:", test_length)

    #     X_hat, _ = jax.vmap(self.trainer.learner.neuralode, in_axes=(0, None, 0, 0))(X[:, :, 0, :], 
    #                                          t_test, 
    #                                          self.trainer.learner.contexts.params,
    #                                          self.trainer.learner.contexts.params)      ## Reuse one's params for testing !

    #     batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

    #     # return batched_criterion(X_hat, X).mean(axis=1).sum(axis=0)
    #     return batched_criterion(X_hat, X).mean(axis=1).mean(axis=0)


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

        X_hat, _ = jax.vmap(self.trainer.learner.neuralode, in_axes=(0, None, 0, 0))(X[:, :, 0, :], 
                                            t_test, 
                                            contexts,
                                            contexts)

        batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

        # crit = batched_criterion(X_hat, X).mean(axis=1).mean(axis=0)
        crit_all = batched_criterion(X, X_hat).mean(axis=1)
        crit = crit_all.mean(axis=0)

        if verbose == True:
            if data_loader.adaptation == False:
                print("Test Score (In-Domain):", crit)
            else:
                print("Test Score (OOD):", crit)
            print(flush=True)

        return crit, crit_all


    # def visualise_cf(self, e=None, traj=None, int_cutoff=1.0, save_path=False, key=None):

    #     e_key, traj_key = get_new_key(time.time_ns(), num=2)
    #     e = e if e else jax.random.randint(e_key, (1,), 0, self.dataloader.nb_envs)[0]
    #     traj = traj if traj else jax.random.randint(traj_key, (1,), 0, self.dataloader.nb_trajs_per_env)[0]

    #     t_eval = self.dataloader.t_eval
    #     test_length = int(self.dataloader.nb_steps_per_traj*int_cutoff)
    #     X = self.dataloader.dataset[e, traj:traj+1, :test_length, :]
    #     t_test = t_eval[:test_length]

    #     print("==  Begining visualisation ... ==")
    #     print("    Environment id:", e)
    #     print("    Trajectory id:", traj)
    #     print("    Length of the training trajectories:", self.trainer.dataloader.int_cutoff)
    #     print("    Length of the testing trajectories:", test_length)

    #     X_hat, _ = self.trainer.learner.neuralode(X[:, 0, :], 
    #                                          t_test, 
    #                                          self.trainer.learner.contexts.params[e],
    #                                          self.trainer.learner.contexts.params[e])   ## TODO addition for NCF

    #     X_hat = X_hat.squeeze()
    #     X = X.squeeze()



    #     fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

    #     mks = 2

    #     ax['A'].plot(t_test, X[:, 0], c="deepskyblue", label=r"$\theta$ (GT)")
    #     ax['A'].plot(t_test, X_hat[:, 0], "o", c="royalblue", label=r"$\theta$ (NODE)", markersize=mks)

    #     ax['A'].plot(t_test, X[:, 1], c="violet", label=r"$\dot \theta$ (GT)")
    #     ax['A'].plot(t_test, X_hat[:, 1], "x", c="purple", label=r"$\dot \theta$ (NODE)", markersize=mks)

    #     ax['A'].set_xlabel("Time")
    #     ax['A'].set_ylabel("State")
    #     ax['A'].set_title("Trajectories")
    #     ax['A'].legend()

    #     ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
    #     ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="Neural ODE")
    #     ax['B'].set_xlabel(r"$\theta$")
    #     ax['B'].set_ylabel(r"$\dot \theta$")
    #     ax['B'].set_title("Phase space")
    #     ax['B'].legend()

    #     nb_steps = np.concatenate(self.trainer.nb_steps_node)
    #     xis = self.trainer.learner.contexts.params
    #     losses_node = np.vstack(self.trainer.losses_node)
    #     losses_ctx = np.vstack(self.trainer.losses_ctx)
    #     nb_envs = self.dataloader.nb_envs

    #     print("    Number of steps taken per epoch:", nb_steps.shape)

    #     mke = np.ceil(losses_node.shape[0]/100).astype(int)

    #     ax['C'].plot(losses_node[:,0], label="NodeLoss", color="grey", linewidth=3, alpha=1.0)
    #     ax['C'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label="ContextLoss", color="grey", linewidth=1, alpha=0.5)
    #     ax['C'].set_xlabel("Epochs")
    #     ax['C'].set_title("Loss Terms")
    #     ax['C'].set_yscale('log')
    #     ax['C'].legend()

    #     ax['D'].plot(nb_steps, c="brown")
    #     ax['D'].set_xlabel("Epochs")
    #     ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs)")
    #     if np.all(nb_steps>0):
    #         ax['D'].set_yscale('log')

    #     eps = 0.1
    #     colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
    #     colors = colors*(nb_envs)

    #     ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
    #     for i, (x, y) in enumerate(xis[:, :2]):
    #         ax['F'].annotate(str(i), (x, y), fontsize=8)
    #     ax['F'].set_title(r'Final Contexts ($\xi^e$)')

    #     init_xis = self.trainer.learner.init_ctx_params
    #     ax['E'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X')
    #     ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
    #     for i, (x, y) in enumerate(init_xis[:, :2]):
    #         ax['E'].annotate(str(i), (x, y), fontsize=8)
    #     for i, (x, y) in enumerate(xis[:, :2]):
    #         ax['F'].annotate(str(i), (x, y), fontsize=8)
    #     ax['E'].set_title(r'Initial Contexts (first 2 dims)')
    #     ax['F'].set_title(r'Final Contexts (first 2 dims)')

    #     plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

    #     plt.tight_layout()
    #     plt.show();

    #     if save_path:
    #         plt.savefig(save_path, dpi=100, bbox_inches='tight')
    #         print("Testing finished. Ffigure saved in:", save_path);

    #     # return fig, ax








    def visualize(self, data_loader, e=None, traj=None, dims=(0,1), context_dims=(0,1), int_cutoff=1.0, save_path=False, key=None):

        # assert data_loader.nb_envs == self.trainer.dataloader.nb_envs, "The number of environments in the test dataloader must be the same as the number of environments in the trainer."

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
        X_hat, _ = self.trainer.learner.neuralode(X[:, 0, :],
                                            t_test, 
                                            contexts[e],
                                            contexts[e])

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        ## Save X_hat in the savepath
        # np.save(save_path+'X_hat.npy', X_hat)

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
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);



    def visualizeUQ(self, data_loader, e=None, traj=None, dims=(0,1), std_scale=1e2, int_cutoff=1.0, save_path=False, forecast=True, verbose=True, title=True, key=None):
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
            t_span_ext = (t_eval[0], t_eval[-1] + (t_eval[-1]-t_eval[0])/2)
            t_test_ext = jnp.linspace(*t_span_ext, 2*test_length)
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
            contexts = jnp.stack([contexts_adapt, contexts], axis=0)

        batched_neuralode = jax.vmap(self.trainer.learner.neuralode, in_axes=(None, None, None, 0))

        X_hat, _ = batched_neuralode(X[:, 0, :], 
                                     t_test_ext, 
                                     contexts[e], 
                                     contexts)

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        # fig, ax = plt.subplot_mosaic('AB', figsize=(6*2, 4*1))
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

        ## If forecasting, the put a vertical line to show when the forecast starts
        if forecast == True:
            ax['A'].axvline(x=t_eval[-1], color='crimson', linestyle='--', label="Forecast Start")

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        # ax['A'].set_title("Trajectories with UQ")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, dim0], X[:, dim1], ".", c="teal", label="GT")
        ax['B'].plot(X_hat_mean[:, dim0], X_hat_mean[:, dim1], c="turquoise", label="NCF")
        ax['B'].fill_between(X_hat_mean[:, dim0], X_hat_mean[:, dim1]-X_hat_std[:, dim1], X_hat_mean[:, dim1]+X_hat_std[:, dim1], color="turquoise", alpha=0.2)

        ax['B'].set_xlabel(f"$x_{{{dim0}}}$")
        ax['B'].set_ylabel(f"$x_{{{dim1}}}$")
        # ax['B'].set_title("Phase space with UQ")
        ax['B'].set_title("Phase space")
        ax['B'].legend()

        if title:
            plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);





    def visualize2D(self, data_loader, e=None, traj=None, res=(32,32), int_cutoff=1.0, nb_plot_timesteps=10, save_path=False, key=None):

        """
        The visualize2D function is used to visualize the results of a trained neural ODE model.
        
        :param self: Access the trainer object
        :param data_loader: Get the data from the dataset
        :param e: Select the environment to visualize
        :param traj: Specify which trajectory to visualize
        :param res: Specify the resolution of the gif
        :param 32): Set the resolution of the gif
        :param int_cutoff: Specify the length of the trajectory to be visualized
        :param nb_plot_timesteps: Specify the number of timesteps to be visualized
        :param save_path: Specify the path where to save the figure
        :param key: Generate a random key for the jax
        :return: A figure with two subplots
        :doc-author: Trelent
        """
        e_key, traj_key = get_new_key(time.time_ns(), num=2)
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
        X_hat, _ = self.trainer.learner.neuralode(X[:, 0, :],
                                            t_test, 
                                            contexts[e],
                                            contexts[e])

        X_hat = X_hat.squeeze()
        X = X.squeeze()

        # if isinstance(res, int):
        #     res = (res, res)
        nb_mats = X_hat.shape[1] // (res*res)
        assert nb_mats > 0, f"Not enough dimensions to form a {res}x{res} matrix"
        # mats = vec_to_mats(X_hat, res, nb_mats)

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
                ax[2*i, j].imshow(gt_j[i], cmap='gist_ncar', interpolation='bilinear', origin='lower')
                ax[2*i+1, j].imshow(ncf_j[i], cmap='gist_ncar', interpolation='bilinear', origin='lower')

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
            print("Testing finished. Figure saved in:", save_path);

        ## Save the gifs as well