"""Command line entry point.

    python -m gnn_aoi download                 # fetch the real datasets (~120 MB) into data/raw/
    python -m gnn_aoi serve                    # live demo website on http://localhost:8000
    python -m gnn_aoi live [--dataset pems_bay] # watch the schedulers run on real data in the terminal
    python -m gnn_aoi train                    # train the GNN scheduler (~40 min on a laptop CPU)
    python -m gnn_aoi experiments [names...]   # run experiments -> results/data/*.json
    python -m gnn_aoi figures                  # results/data -> results/figures/*.png
    python -m gnn_aoi animate                  # hero.png + scheduler.gif
    python -m gnn_aoi demo                     # quick side-by-side on one layout (seconds)
    python -m gnn_aoi all                      # everything after training
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser(prog="gnn_aoi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=None, help="default: $PORT or 8000")
    sv.add_argument("--host", default=None, help="default: 0.0.0.0 when $PORT is set, else 127.0.0.1")
    lv = sub.add_parser("live")
    lv.add_argument("--dataset", default="intel_lab", choices=["intel_lab", "pems_bay"])
    lv.add_argument("--slots", type=int, default=120)
    lv.add_argument("--speed", type=float, default=4.0, help="slots per second")
    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--layers", type=int, default=3, help="message-passing layers (0 = ablation)")
    t.add_argument("--threads", type=int, default=8)
    t.add_argument("--out", default="checkpoints/gnn_scheduler.pt")
    t.add_argument("--log", default="results/data/training_log.json")
    t.add_argument("--init-from", default="", help="warm-start the actor from a checkpoint")
    e = sub.add_parser("experiments")
    e.add_argument("names", nargs="*", help="subset of experiments (default: all)")
    sub.add_parser("figures")
    sub.add_parser("animate")
    d = sub.add_parser("demo")
    d.add_argument("--layout", default="clustered", choices=["uniform", "clustered"])
    d.add_argument("--seed", type=int, default=10_003)
    sub.add_parser("all")
    args = ap.parse_args()

    if args.cmd == "download":
        from gnn_aoi.datasets import download
        download()

    elif args.cmd == "serve":
        from gnn_aoi.server import serve
        serve(args.port, args.host)

    elif args.cmd == "live":
        from gnn_aoi.live import terminal
        terminal(args.dataset, args.slots, args.speed)

    elif args.cmd == "train":
        from gnn_aoi.gnn.train import TrainConfig, train
        train(TrainConfig(ppo_iters=args.iters, seed=args.seed, layers=args.layers, threads=args.threads,
                          out=args.out, log=args.log, init_from=args.init_from))

    elif args.cmd in ("experiments", "all"):
        from gnn_aoi.experiments import ALL
        names = getattr(args, "names", None) or list(ALL)
        for n in names:
            t0 = time.time()
            print(f"▶ {n}", flush=True)
            ALL[n]()
            print(f"  ✓ {n} ({time.time() - t0:.0f}s)", flush=True)

    if args.cmd in ("figures", "all"):
        from gnn_aoi.viz.plots import all_figures
        all_figures()

    if args.cmd in ("animate", "all"):
        from gnn_aoi.viz.animate import render
        render()

    if args.cmd == "demo":
        from gnn_aoi.env import SchedulingEnv
        from gnn_aoi.evaluate import run_episode
        from gnn_aoi.experiments import HEADLINE, policies
        sc = HEADLINE.with_(layout=args.layout)
        print(f"Synthetic {args.layout} layout, M={sc.M}, C={sc.C}, AoI <= {sc.freshness}")
        print(f"{'policy':<26}{'MSE':>9}{'mean AoI':>10}{'peak AoI':>10}{'ms/slot':>9}")
        for pol in policies():
            _, s = run_episode(SchedulingEnv(sc, layout_seed=args.seed), pol, T=300)
            print(f"{pol.label:<26}{s['mse']:>9.4f}{s['mean_age']:>10.2f}{s['max_age']:>10.0f}{s['ms_per_slot']:>9.3f}")


if __name__ == "__main__":
    main()
