module OutbreakModel
    export Parameters, simulate_fatality_curves, distance_name, earth_movers_distance, max_distance, mean_distance

    using Distributions

    abstract type Parameters end

    struct BinomialParameters <: Parameters
        α::Float64
        p::Float64
    end

    struct WeibullParameters <: Parameters
        k::Float64
        l::Float64
    end

    function fatality_binomial(params::BinomialParameters, n::Int64, k::Int64)
        return Binomial(n, params.α * (1 - (1 - params.p)^k))
    end

    function fatality_weibull(params::Parameters)

    function simulate_fatality_curves(params::Parameters, epidemic_curve::Array{Int64}, nsims::Int64 = 10_000)
        T = size(epidemic_curve, 1)
        simulations = Array{Int64, 2}(undef, nsims, T)

        for (t, n) in enumerate(epidemic_curve)
            D_t = fatality_binomial(params, n, T - t)

            simulations[:, t] = rand(D_t, nsims)
        end

        return simulations
    end

    using Distributed
    using SharedArrays

    # from https://github.com/mirkobunse/EarthMoversDistance.jl (needs citation)
    @everywhere using EarthMoversDistance
    @everywhere using LinearAlgebra: normalize
    @everywhere using Distances

    function distance_name(distance::Function)
        if distance == earth_movers_distance
            return "emd"
        elseif distance == mean_distance
            return "mean"
        elseif distance == max_distance
            return "max"
        end

        error("Unknown distance functio")
    end

    function earth_movers_distance(fatality_curve::Array{Int64}, fatality_curve_simulations::Array{Int64, 2})
        nsims, T = size(fatality_curve_simulations)

        fatality_histogram = normalize(fatality_curve, 1)
        cumulative_fatality_curve = cumsum(fatality_curve)

        min_emd = @distributed (min) for k in 1:nsims
            fatality_curve_sim = fatality_curve_simulations[k, :]

            if sum(fatality_curve_sim) == 0
                return NaN
            end

            simulation_histogram = normalize(fatality_curve_sim, 1)

            emd = earthmovers(fatality_histogram, simulation_histogram, Cityblock())
            penalty = abs(cumulative_fatality_curve[T] - sum(fatality_curve_sim)) / cumulative_fatality_curve[T]

            emd + penalty
        end

        return min_emd
    end

    function max_distance(fatality_curve::Array{Int64}, fatality_curve_simulations::Array{Int64, 2})
        T = size(fatality_curve, 1)

        cumulative_fatality_curve = cumsum(fatality_curve)
        cumulative_fatality_curve_simulations = cumsum(fatality_curve_simulations, dims=2)

        max_distance = @distributed (max) for k in 1:T
            abs(cumulative_fatality_curve[k] + mean(cumulative_fatality_curve_simulations[:, k]))
        end

        # distances = zeros(T)
        #
        # for k in 1:T
        #     distances[k] = abs(cumulative_fatality_curve[k] + mean(cumulative_fatality_curve_simulations[:, k]))
        # end

        return max_distance
    end

    function mean_distance(fatality_curve::Array{Int64}, fatality_curve_simulations::Array{Int64, 2})
        nsims, T = size(fatality_curve_simulations)

        cumulative_fatality_curve = cumsum(fatality_curve)
        cumulative_fatality_curve_simulations = cumsum(fatality_curve_simulations, dims=2)

        total_distances = @distributed (+) for k in 1:T
            abs(cumulative_fatality_curve[k] + mean(cumulative_fatality_curve_simulations[:, k]))
        end

        return total_distances / T
    end
end  # module ModelOptimization
