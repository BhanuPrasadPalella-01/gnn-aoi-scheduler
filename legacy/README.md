# Legacy (mid-semester) code

The original exploratory scripts, kept for reference. They are superseded by the `gnn_aoi` package.

**Known issue:** these scripts estimate the correlation graph on a simulator built with `seed=42`
but evaluate on `seed=123`. The seed also draws the sensor positions, so the graph describes a
different set of sensors (r = 0.04 against the evaluated layout's true correlation). The
"correlation-aware" numbers produced here are therefore not meaningful. The new package separates
`layout_seed` from the noise seed; `tests/test_core.py::test_env_graph_describes_its_own_sensors`
guards against regressing.

To run a legacy script: `cd legacy && PYTHONPATH=. ../venv/bin/python scripts/run_b1.py`
