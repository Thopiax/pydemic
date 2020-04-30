module ModelOptimization
    export ModelConfiguration, gridsearch

    using ..OutbreakData
    using ..OutbreakModel
    using DataFrames, DataStructures, Distributed, CSV

    struct ModelConfiguration
        timeline::OutbreakTimeline
        α_space::StepRangeLen
        p_space::StepRangeLen
        distance_metric::Function
        ϵ::Float64
        recompute::Bool
    end

    function prune_inadmissible!(df::DataFrame, threshold::Float64, key=:d)
        filter!(row -> row[key] <= threshold, df);
    end

    function build_filepath(modelconfig::ModelConfiguration)
        admissible_percentage = round(Int, modelconfig.ϵ * 100)

        path = joinpath(@__DIR__, "../results/$(modelconfig.timeline.name)/")
        if !isdir(path) mkdir(path) end

        filename = "admissible_set_$(modelconfig.timeline.configname)_$(admissible_percentage)_$(distance_name(modelconfig.distance_metric)).csv"

        return joinpath(path, filename)
    end

    function save_result(modelconfig::ModelConfiguration, df::DataFrame)
        filepath = build_filepath(modelconfig)

        CSV.write(filepath, df)
    end

    function load_result(modelconfig::ModelConfiguration)
        filepath = build_filepath(modelconfig)

        if !modelconfig.recompute && isfile(filepath)
            return CSV.read(filepath)
        end

        return nothing
    end

    function gridsearch(modelconfig::ModelConfiguration, verbose::Bool = true)
        prev_result = load_result(modelconfig)

        if !isnothing(prev_result)
            @warn "Using a previous result."
            return prev_result
        end

        admissible_df = DataFrame(OrderedDict("α" => [],"p" => [], "d" => []))

        minimum_distance = nothing
        distance_threshold = nothing

        for α in modelconfig.α_space
            for p in modelconfig.p_space
                parameters = Parameters(α, p)
                simulations = simulate_fatality_curves(parameters, modelconfig.timeline.epidemic_curve)

                distance = modelconfig.distance_metric(modelconfig.timeline.fatality_curve, simulations)

                if isnan(distance) continue end

                if isnothing(minimum_distance) || distance <= minimum_distance
                    minimum_distance = distance
                    distance_threshold = (1 + modelconfig.ϵ) * distance

                    prune_inadmissible!(admissible_df, distance_threshold)

                    @warn "New Minimum" α p minimum_distance distance_threshold
                end

                if distance <= distance_threshold
                    push!(admissible_df, [α, p, distance])

                    @info "Admissible Parameters" α p distance S=size(admissible_df, 1)
                end
            end
        end

        save_result(modelconfig, admissible_df)

        return admissible_df
    end

end
