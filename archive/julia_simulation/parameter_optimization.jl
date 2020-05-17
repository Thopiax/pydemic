module PandemicParameterOptimization
    export ModelConfiguration

    using ..PandemicSimulation, ..Pandemic
    using BlackBoxOptim, BayesianOptimization, GaussianProcesses, Distributions

    struct ModelConfiguration
        outbreak::Outbreak
        searchspace::Array{Tuple{Float64, Float64}, 1}
        nsims::Int64
        distance_metric::Function

        ModelConfiguration(outbreak, searchspace) = new(
            outbreak, searchspace, 10_000, mean_distance
        )

        ModelConfiguration(outbreak, searchspace, nsims) = new(
            outbreak, searchspace, nsims, mean_distance
        )

        ModelConfiguration(outbreak, searchspace, nsims, distance_metric) = new(
            outbreak, searchspace, nsims, distance_metric
        )
    end

    function blackboxopt(config::ModelConfiguration)
        evaluator = build_evaluator(config)

        return bboptimize(evaluator; SearchRange=config.searchspace,  Method = :random_search)
    end

    function run!(optimizer)
        return boptimize!(optimizer)
    end

    function build_optimizer(config::ModelConfiguration, niters::Int)
        model = ElasticGPE(3, capacity = 3000)

        return BOpt(
            build_evaluator(config),  # f
            model,                    # model
            UpperConfidenceBound(),    # acquisition
            NoModelOptimizer(),       # optimizer
            config.searchspace[:, 1], # lowerbound
            config.searchspace[:, 2], # upperbound
            maxiterations = niters,
            initializer =
            sense = Min,
            verbosity = Timings
        )
    end

    function build_evaluator(config::ModelConfiguration)
        function _evaluator(x)
            simparams = PandemicSimulation.Parameters(x[1], x[2], x[3])
            fatality_simulations = PandemicSimulation.simulate_outbreak(simparams, config.outbreak.epidemic_curve, config.nsims)

            return config.distance_metric(config.outbreak.fatality_curve, fatality_simulations)
        end

        return _evaluator
    end
end
