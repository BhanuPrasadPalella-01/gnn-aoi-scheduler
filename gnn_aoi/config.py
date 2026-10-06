from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class Scenario:
    """Everything that defines one scheduling problem instance.

    M sensors sit in the unit square; their readings form a spatially
    correlated (RBF kernel) field that evolves in time as an AR(1) process.
    A sink can poll at most C sensors per slot and must keep an estimate of
    *all* M readings.
    """
    M: int = 20                      # number of sensors
    C: int = 5                       # polls per slot (channel capacity)
    length_scale: float = 0.3        # RBF spatial correlation length
    ar_rho: float = 0.9              # temporal correlation per slot
    rho_spread: float = 0.0          # per-sensor rho drawn from ar_rho +/- rho_spread
    layout: str = "uniform"          # "uniform" | "clustered"
    n_clusters: int = 4
    cluster_std: float = 0.07
    obs_noise_var: float = 1e-3      # sensor measurement noise variance
    interference_radius: float = 0.0 # >0 => sensors this close can't share a slot
    pilot_len: int = 1000            # slots of full observation used to estimate the correlation
    corr_threshold: float = 0.3      # |corr| >= threshold => edge in the correlation graph
    freshness: int = 0               # >0 => no sensor may exceed this age (forced polls); 0 = off

    def with_(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)
