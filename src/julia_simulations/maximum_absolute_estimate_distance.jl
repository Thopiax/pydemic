using DataFrames
using CSV
using Distributions
using DataStructures

function fatality_distribution(x, k, α, p)
    n = round(Int, x)
    return Binomial(n, α * (1 - (1 - p)^k))
end

function fatality_trace_samples(E, nsamples, α, p)
    T = size(E, 1)
    ans = Array{Float64, 2}(undef, nsamples, T)

    for (t, n) in enumerate(E)
        D_t = fatality_distribution(n, T - t, α, p)

        ans[:, t] = rand(D_t, nsamples)
    end

    return ans
end

function cumulative_fatality_samples(E, nsamples, α, p)
    ans = fatality_trace_samples(E, nsamples, α, p)

    return vec(sum(ans, dims=2))
end


function filterparams!(param_df, threshold)
    filter!(row -> row[:d] <= threshold, param_df);
end

# from https://github.com/mirkobunse/EarthMoversDistance.jl (needs citation)
using EarthMoversDistance
using LinearAlgebra
using Distances

function earth_movers_distance(E, X, T, nsims, α, p)
    fatality_traces = fatality_trace_samples(E, nsims, α, p)

    X_hist = LinearAlgebra.normalize(X, 1)

    distances = zeros(nsims)

    for (i, trace) in enumerate(eachrow(fatality_traces))
        if sum(trace) == 0
            return nothing
        end

        trace_hist = LinearAlgebra.normalize(trace, 1)

        trace_emd = earthmovers(X_hist, trace_hist, Cityblock())
        trace_diff = abs(sum(X) - sum(trace)) / sum(X)

        distances[i] = trace_emd
    end

    return minimum(distances)
end

function max_distance(E, X, T, nsims, α, p)
    maximum_distance = -1
    for k in 1:T
        total_fatality_samples = cumulative_fatality_samples(E[1:k], nsims, α, p)
        expected_total_fatalities = mean(total_fatality_samples)

        actual_distance = abs(X[k] - expected_total_fatalities)

        if actual_distance > maximum_distance
            ans = actual_distance
        end
    end

    return maximum_distance
end

function mean_distance(E, X, T, nsims, α, p)
    distances = []
    for k in 1:T
        total_fatality_samples = cumulative_fatality_samples(E[1:k], nsims, α, p)
        expected_total_fatalities = mean(total_fatality_samples)

        push!(distances, abs(X[k] - expected_total_fatalities))
    end

    return mean(distances)
end

function choose_metric(metric)
    if metric == "mean"
        return mean_distance
    elseif metric == "max"
        return max_distance
    elseif metric == "emd"
        return earth_movers_distance
    end

    # defaults to mean distance
    return mean_distance
end

# Returns the day where the Xth case occured
function FFXidx(X, E)
    sum = 0

    for (i, e) in enumerate(E)
        sum += e

        if sum >= X
            return i
        end
    end

    return size(E, 1)
end

function new_parameter_search(name, T, α_space, p_space, admissibility_window=0.1, FFX=false, distance_metric="mean", nsims=10000, verbose=true)
    df = "./data/time_series/diff/$name.csv" |> CSV.File |> DataFrame

    if FFX
        X = T
        T = FFXidx(X, df.Infected)

        censor_string = "$(X)FFX"
        println("Case #$X occured on day $T after infection")
    else
        censor_string = "$(T)d"
    end

    @assert 0 < T <= size(df, 1)
    @assert 0 < admissibility_window < 1

    E = copy(df.Infected[1:T]) # epidemic curve
    X = cumsum(df.Dead)[1:T] # cumulative mortality curve

    param_df = DataFrame(OrderedDict("α" => [],"p" => [], "d" => []))
    minimum_distance = nothing

    for α in α_space
        println("[$name] Running for α=$α")
        for p in p_space
            distance = choose_metric(distance_metric)(E, X, T, nsims, α, p)

            # if distance is nothing then the parameters were not valid and resulted in a 0 vector.
            if isnothing(distance)
                continue
            end

            if isnothing(minimum_distance) || distance <= minimum_distance
                minimum_distance = distance

                if verbose
                    println("[$name] (α=$α, p=$p) New minimum distance $minimum_distance")
                end

                filterparams!(param_df, (1 + admissibility_window) * minimum_distance)
            end

            if distance <= (1 + admissibility_window) * minimum_distance
                push!(param_df, [α, p, distance])

                if verbose
                    println("[$name] (α=$α, p=$p) Admissible Parameter
                            distance=$(distance)
                            threshold=$((1 + admissibility_window) * minimum_distance)
                            size(param_df)=$(size(param_df, 1))
                    ")
                end
            end
        end
    end

    println("[$name] Writing parameters to file.")

    param_df |> CSV.write("./results/$(name)_$(censor_string)_$(round(Int, admissibility_window * 100))_$(distance_metric)_params.csv")

    return param_df
end


SARS_α_space = 0.05:0.001:0.15
SARS_p_space = 0.01:0.01:1

new_parameter_search("SARS", 100, SARS_α_space, SARS_p_space, 0.2, true, "emd")
new_parameter_search("SARS", 1000, SARS_α_space, SARS_p_space, 0.2, true, "emd")
new_parameter_search("SARS", 5000, SARS_α_space, SARS_p_space, 0.2, true)
new_parameter_search("SARS", 10000, SARS_α_space, SARS_p_space, 0.2, true)

# MERS

MERS_α_space = 0.20:0.01:0.30
MERS_p_space = 0:0.02:1

new_parameter_search("MERS_first", 100, MERS_α_space, MERS_p_space, 0.2, true)
new_parameter_search("MERS_first", 1000, MERS_α_space, MERS_p_space, 0.2, true)

new_parameter_search("MERS_second", 100, MERS_α_space, MERS_p_space, 0.2, true)
new_parameter_search("MERS_second", 1000, MERS_α_space, MERS_p_space, 0.2, true)

# Coronavirus

Corona_α_space = 0.01:0.001:0.1
Corona_p_space = 0:0.02:1

new_parameter_search("Coronavirus", 100, Corona_α_space, Corona_p_space, 0.2, true)
new_parameter_search("Coronavirus", 1_000, Corona_α_space, Corona_p_space, 0.2, true)
new_parameter_search("Coronavirus", 10_000, Corona_α_space, Corona_p_space, 0.2, true)
new_parameter_search("Coronavirus", 100_000, Corona_α_space, Corona_p_space, 0.2, true)
