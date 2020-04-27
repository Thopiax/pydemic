include("utils.jl");

# # Add 4 workers when this model is loaded
using Distributed;
addprocs(4);
nworkers()

include_everywhere("data.jl");
include_everywhere("model.jl");
include_everywhere("optimization.jl");

# iterate through all countries
itr = walkdir(joinpath(@__DIR__, "../data/countries"));
filename_pattern = r"([^.]+).csv";

(root, _, coronavirus_country_files) = first(itr);

for country_name in ["US", "Spain", "Italy", "France", "Germany", "United Kingdom", "Turkey", "China", "Russia", "Brazil"]
    country_file = "$country_name.csv"
    # country_name = convert(String, match(filename_pattern, country_file)[1])

    ts = OutbreakData.load(country_name, joinpath(root, country_file))

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
