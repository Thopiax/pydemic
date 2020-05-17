include("utils.jl");

# # Add 4 workers when this model is loaded
using Distributed;
addprocs(4);
nworkers()

include_everywhere("pandemic.jl");
include_everywhere("simulation.jl");
include_everywhere("parameter_optimization.jl");

# iterate through all countries
itr = walkdir(joinpath(@__DIR__, "../data/coronavirus"));
(root, _, coronavirus_country_files) = first(itr);
filename_pattern = r"([^.]+).csv";

outbreak = Pandemic.load_outbreak("Italy", joinpath(root, "Italy.csv"))

config = PandemicParameterOptimization.ModelConfiguration(outbreak, [(0.000001, 1.0), (1.0, 2.2), (5.0, 20.0)], 1_000)

res = PandemicParameterOptimization.blackboxopt(config)

sims = PandemicSimulation.simulate_outbreak(PandemicSimulation.Parameters(0.014686853425672155, 2.028522724987662, 16.289399712844542), outbreak.epidemic_curve, 1000)

PandemicSimulation.mean_distance(outbreak.fatality_curve, sims)

for country_name in ["US", "Spain", "Italy", "France", "Germany", "United Kingdom", "Turkey", "China", "Russia", "Brazil", "Iran"]
    # use smoothed diffs
    country_file = "$(country_name)_s3.csv"

    outbreak = Pandemic.load_outbreak(country_name, joinpath(root, country_file))

    for d in 2:5
        X = 10^d

        censored_ts = OutbreakData.censorbyffx(ts, X)

        if sum(censored_ts.fatality_curve) == 0
            @info "Skipping due to no fatalities." country_name X
            continue
        end

        for distance_metric in [OutbreakModel.mean_distance, OutbreakModel.earth_movers_distance]
            @info "Starting grid search" country_name X distance_metric=OutbreakModel.distance_name(distance_metric)

            admissible_df = ModelOptimization.gridsearch(ModelOptimization.ModelConfiguration(
                censored_ts,
                0.01:0.001:0.13,
                0.2:0.01:0.8,
                distance_metric,
                0.2,
                false
            ))

            @info "Finished grid search" best_d=minimum(admissible_df.d) best_params=admissible_df[argmin(admissible_df.d), [:α, :p]]
        end
    end
end
