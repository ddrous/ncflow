from ._utils import *

class DataLoader:
    def __init__(self, dataset, t_eval=None, batch_size=-1, int_cutoff=0.2, shuffle=True, key=None, adaptation=False):

        if isinstance(dataset, str):
            raw_dat = jnp.load(dataset)
            self.dataset, self.t_eval = jnp.asarray(raw_dat['X']), jnp.asarray(raw_dat['t'])
        else:
            self.dataset = dataset
            self.t_eval = t_eval

        self.shuffle = shuffle
        if self.shuffle:
            if key is None:
                print("WARNING: You are demanding a shuffled dataset but did not provide any keys for that.")
                self.key = get_new_key()
            else:
                self.key = key

        print("Dataset type:", type(self.dataset))

        assert jnp.ndim(self.dataset) == 4, "Dataset must be of shape (nb_envs, nb_trajs_per_env, nb_steps_per_traj, data_size)"
        assert self.t_eval.shape[0] == self.dataset.shape[2], "t_eval must have the same length as the number of steps in the dataset"

        datashape = self.dataset.shape
        self.nb_envs = datashape[0]
        self.nb_trajs_per_env = datashape[1]
        self.nb_steps_per_traj = datashape[2]
        self.data_size = datashape[3]

        print("Dataset shape:", datashape)

        self.int_cutoff = int(int_cutoff*self.nb_steps_per_traj)    ## integration cutoff

        if batch_size < 0 or batch_size > self.nb_trajs_per_env:
            print("WARNING: batch_size must be between 0 and nb_trajs_per_env. Setting batch_size to maximum.")
            self.batch_size = self.nb_trajs_per_env
        else:
            self.batch_size = batch_size

        self.adaptation = adaptation    ## Is this a dataset for adaptation ?

    # def __iter__(self):     ## TODO! Randomise this function
    #     nb_batches = self.nb_trajs_per_env // self.batch_size
    #     for batch_id in range(nb_batches):
    #         traj_start, traj_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size
    #         yield self.dataset[:, traj_start:traj_end, :self.int_cutoff, :], self.t_eval[:self.int_cutoff]

    def __iter__(self):
        nb_batches = self.nb_trajs_per_env // self.batch_size
        key = get_new_key(self.key)
        perm_dataset = self.dataset

        if self.shuffle:
            ## The strategy below eleviates encountering the same (env1, traj1) - (env2, traj2) pair across all batches

            ## 1) Extract a subset of environments
            e_start = jax.random.randint(key, shape=(1,), minval=0, maxval=self.nb_envs)[0]
            length = jax.random.randint(key, shape=(1,), minval=e_start+1, maxval=self.nb_envs+1)[0] - e_start
            ## 2) Shuffle that subset accross dimension 1 (trajs), then put them back at the same place
            perm_env = jax.random.permutation(key, self.dataset[e_start:e_start+length, ...], axis=1)
            perm_dataset = self.dataset.at[e_start:e_start+length, ...].set(perm_env)
            ## 3) Shuffle the resulting dataset again accross dimension 1 (for extra randomness)
            perm_dataset = jax.random.permutation(key, perm_dataset, axis=1)

            # ## 1) Extract a subset of environments
            # e_start = jax.random.randint(key, shape=(1,), minval=0, maxval=self.nb_envs)[0]
            # length = jax.random.randint(key, shape=(1,), minval=e_start+1, maxval=self.nb_envs+1)[0] - e_start
            # ## 2) Shuffle that subset accross dimension 1 (trajs), then put them back at the same place
            # perm_env = jax.random.permutation(key, perm_dataset[e_start:e_start+length, ...], axis=1)
            # perm_dataset = self.dataset.at[e_start:e_start+length, ...].set(perm_env)
            # ## 3) Shuffle the resulting dataset again accross dimension 1 (for extra randomness)
            # perm_dataset = jax.random.permutation(key, perm_dataset, axis=1)

        else:
            perm_dataset = self.dataset

        ## We are now ready to iterate over the dataset
        for batch_id in range(nb_batches):
            traj_start, traj_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size
            yield perm_dataset[:, traj_start:traj_end, :self.int_cutoff, :], self.t_eval[:self.int_cutoff]

        self.key = key

    def __len__(self):
        return self.nb_envs * self.nb_trajs_per_env
