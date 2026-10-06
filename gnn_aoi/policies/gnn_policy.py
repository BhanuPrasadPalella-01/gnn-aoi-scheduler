import torch

from gnn_aoi.policies.baselines import Policy


class GNNPolicy(Policy):
    """Deployable learned scheduler: greedy decode of the trained actor.
    Uses only what the sink knows (no true covariance)."""
    name, label = "gnn", "GNN-PPO (ours)"

    def __init__(self, checkpoint="checkpoints/gnn_scheduler.pt", name=None, label=None):
        self.checkpoint = checkpoint
        self.actor = None
        if name:
            self.name, self.label = name, label

    def reset(self, env):
        if self.actor is None:
            from gnn_aoi.gnn.train import load_actor
            torch.set_num_threads(1)
            self.actor, _ = load_actor(self.checkpoint)
        self.env = env

    @torch.no_grad()
    def select(self, obs):
        from gnn_aoi.gnn.features import featurize
        from gnn_aoi.gnn.train import full_picks
        picks, _, _ = self.actor.decode(featurize([self.env], [obs]), greedy=True)
        return full_picks(obs, picks[0].numpy())

    def priority(self, obs):
        """Probability the GNN gives each sensor for its first pick this slot
        (call after select)."""
        p = getattr(self.actor, "last_priority", None)
        return None if p is None else p[0].numpy().astype(float)

    def __getstate__(self):  # ship the path to worker processes, not the model
        return {**self.__dict__, "actor": None, "env": None}
