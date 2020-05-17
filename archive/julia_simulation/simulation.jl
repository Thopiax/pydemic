module PandemicSimulation
    export Parameters, simulate_outbreak, distance_name, max_distance, mean_distance

    using Distributions

    struct Parameters
        α::Float64
        k::Float64
        l::Float64
    end

    function simulate_fatalities(rv::Sampleable, ncases::Int64)
        # assumption: a death occuring on a day will be reported the next day.
        return map(x -> ceil(Int, x), rand(rv, ncases))
    end

    function simulate_outbreak(params::Parameters, epidemic_curve::Vector{Int64}, nsims::Int64 = 10_000) :: Array{Int64, 2}
        T = size(epidemic_curve, 1)
        fatality_simulations = Array{Int64, 2}(undef, nsims, T)

        fatality_hazard_rate = Weibull(params.k, params.l)

        for (t, ncases) in enumerate(epidemic_curve)
            casefatalities = rand(Binomial(ncases, params.α), nsims)
            @info "New day" f_t=size(casefatalities, 1) c_t=ncases

            for i in 1:nsims
                fatality_report_delays = simulate_fatalities(fatality_hazard_rate, casefatalities[i])
                @info "Simulation #$(i)"
                censored_fatality_dates = filter(x -> x <= T, t .+ fatality_report_delays)

                for date in censored_fatality_dates
                    fatality_simulations[i, min(date, T)] += 1
                end
            end
        end

        return fatality_simulations
    end

    using Distributed
    using SharedArrays

    function distance_name(distance::Function)
        if distance == mean_distance
            return "mean"
        elseif distance == max_distance
            return "max"
        end

        error("Unknown distance function")
    end

    function max_distance(fatality_curve::Array{Int64}, fatality_curve_simulations::Array{Int64, 2})
        T = size(fatality_curve, 1)

        cumulative_fatality_curve = cumsum(fatality_curve)
        cumulative_fatality_curve_simulations = cumsum(fatality_curve_simulations, dims=2)

        max_distance = @distributed (max) for k in 1:T
            abs(cumulative_fatality_curve[k] + mean(cumulative_fatality_curve_simulations[:, k]))
        end

        return max_distance
    end

    function mean_distance(fatality_curve::Array{Int64}, fatality_curve_simulations::Array{Int64, 2})
        nsims, T = size(fatality_curve_simulations)

        cumulative_fatality_curve = cumsum(fatality_curve)
        mean_cumulative_fatality_curve_simulations = mean(cumsum(fatality_curve_simulations, dims=2), dims=2)

        total_distances = @distributed (+) for i in 1:T
            abs(cumulative_fatality_curve[i] + mean_cumulative_fatality_curve_simulations[i])
        end

        return total_distances / T
    end
end  # module PandemicSimulation
