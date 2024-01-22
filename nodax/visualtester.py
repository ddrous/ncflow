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


    def test(self, data_loader, criterion=None, int_cutoff=1.0):
        """ Compute test metrics on the adaptation dataloader  """

        criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

        t_eval = data_loader.t_eval
        test_length = int(data_loader.nb_steps_per_traj*int_cutoff)
        X = data_loader.dataset[:, :, :test_length, :]
        t_test = t_eval[:test_length]

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

        crit = batched_criterion(X_hat, X).mean(axis=1).mean(axis=0)

        if data_loader.adaptation == False:
            print("Test Score (In-Domain):", crit)
        else:
            print("Test Score (OOD):", crit)
        print()

        return crit


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








    def visualize(self, data_loader, e=None, traj=None, int_cutoff=1.0, save_path=False, key=None):

        # assert data_loader.nb_envs == self.trainer.dataloader.nb_envs, "The number of environments in the test dataloader must be the same as the number of environments in the trainer."

        e_key, traj_key = get_new_key(time.time_ns(), num=2)
        e = e if e else jax.random.randint(e_key, (1,), 0, data_loader.nb_envs)[0]
        traj = traj if traj else jax.random.randint(traj_key, (1,), 0, data_loader.nb_trajs_per_env)[0]

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

        fig, ax = plt.subplot_mosaic('AB;CC;DD;EF', figsize=(6*2, 3.5*4))

        mks = 2

        ax['A'].plot(t_test, X[:, 0], c="deepskyblue", label=r"$\theta$ (GT)")
        ax['A'].plot(t_test, X_hat[:, 0], "o", c="royalblue", label=r"$\theta$ (NODE)", markersize=mks)

        ax['A'].plot(t_test, X[:, 1], c="violet", label=r"$\dot \theta$ (GT)")
        ax['A'].plot(t_test, X_hat[:, 1], "x", c="purple", label=r"$\dot \theta$ (NODE)", markersize=mks)

        ax['A'].set_xlabel("Time")
        ax['A'].set_ylabel("State")
        ax['A'].set_title("Trajectories")
        ax['A'].legend()

        ax['B'].plot(X[:, 0], X[:, 1], c="turquoise", label="GT")
        ax['B'].plot(X_hat[:, 0], X_hat[:, 1], ".", c="teal", label="Neural ODE")
        ax['B'].set_xlabel(r"$\theta$")
        ax['B'].set_ylabel(r"$\dot \theta$")
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
            init_xis = self.trainer.learner.init_ctx_params_adapt.params

        mke = np.ceil(losses_node.shape[0]/100).astype(int)

        label_node = "NodeLoss" if data_loader.adaptation == False else "ContextLossAdapt"
        ax['C'].plot(losses_node[:,0], label=label_node, color="grey", linewidth=3, alpha=1.0)
        label_ctx = "ContextLoss" if data_loader.adaptation == False else "ContextLossAdapt"
        ax['C'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)
        ax['C'].set_xlabel("Epochs")
        ax['C'].set_title("Loss Terms")
        ax['C'].set_yscale('log')
        ax['C'].legend()

        ax['D'].plot(nb_steps, c="brown")
        ax['D'].set_xlabel("Epochs")
        ax['D'].set_title("Total Number of Steps Taken per Epoch (Proportional to NFEs)")
        if np.all(nb_steps>0):
            ax['D'].set_yscale('log')

        eps = 0.1
        colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
        colors = colors*(nb_envs)

        ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
        for i, (x, y) in enumerate(xis[:, :2]):
            ax['F'].annotate(str(i), (x, y), fontsize=8)
        ax['F'].set_title(r'Final Contexts ($\xi^e$)')

        ax['E'].scatter(init_xis[:,0], init_xis[:,1], s=30, c=colors[:nb_envs], marker='X')
        ax['F'].scatter(xis[:,0], xis[:,1], s=50, c=colors[:nb_envs], marker='o')
        for i, (x, y) in enumerate(init_xis[:, :2]):
            ax['E'].annotate(str(i), (x, y), fontsize=8)
        for i, (x, y) in enumerate(xis[:, :2]):
            ax['F'].annotate(str(i), (x, y), fontsize=8)
        ax['E'].set_title(r'Initial Contexts (first 2 dims)')
        ax['F'].set_title(r'Final Contexts (first 2 dims)')

        plt.suptitle(f"Results for env={e}, traj={traj}", fontsize=14)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);
