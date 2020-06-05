from skopt.space import Real

# variables below found from summary statistics in study
WUHAN_BETA = 1.8429282373343958
WUHAN_LAMBDA = 10.018568258846706

DIMENSIONS = {
    "initial": [
        Real(0.01, 0.20),
        Real(0.5, 10.0),
        Real(1.0, 20.0)
    ], "relaxed": [
        Real(0.0, 1.0),
        Real(0.0, 10.0),
        Real(0.0, 50.0)
    ]
}