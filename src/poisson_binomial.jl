using DataFrames
using CSV
using Distributions

function daily_fatality_distribution(n, k, α, p)
    return Binomial(n, α * (1 - (1 - p)^k))
end

function simulate_total_fatality_samples(E, nsamples, α, p)
    T = size(E, 1)
    ans = Array{Float64, 2}(undef, nsamples, T)

    for (t, n) in enumerate(E)
        D_t = daily_fatality_distribution(n, T - t, α, p)

        ans[:, t] = rand(D_t, nsamples)
    end

    return vec(sum(ans, dims=2))
end

function log_likelihood(E, y, α, p)
    T = size(E, 1)
    return sum([pdf.(daily_fatality_distribution(E[k], T - k, α, p), round(Int, y[k])) for k in 1:(T - 1)])
end

function confidence_interval(confidence::Int)::Vector{Float64}
    confidence /= 100
    return [(1 - confidence) / 2, confidence + (1 - confidence) / 2]
end

function calculate_admissible_parameters(name, censor_end, confidence, α_space, p_space)
    df = "./data/time_series/diff/$name.csv" |> CSV.File |> DataFrame

    low_confidence_threshold, high_confidence_threshold = confidence_interval(confidence)

    # E = epidemic curve
    E = df.Infected[1:censor_end]

    # y = real death count
    y = df.Dead[1:censor_end]

    true_deaths = sum(y)

    admissible_parameters = []
    for α in α_space
        for p in p_space
            total_fatality_samples = simulate_total_fatality_samples(E, 10000, α, p)

            println("[$name] (α=$α, p=$p) Total Fatality samples: μ=$(mean(total_fatality_samples)), σ^2=$(var(total_fatality_samples))")

            low_deaths, high_deaths = quantile(total_fatality_samples, [low_confidence_threshold,  high_confidence_threshold])

            if low_deaths <= true_deaths <= high_deaths
                l = log_likelihood(E, y, α, p)
                println("[$name] (α=$α, p=$p) Likelihood=$l")
                println("[$name] (α=$α, p=$p) Parameters are admissible! [$low_deaths, $high_deaths]")
                push!(admissible_parameters, (α, p, l))
            end
        end
    end

    println("[$name] Writing parameters to file.")

    rename(DataFrame(admissible_parameters), [1 => "α", 2 => "p", 3 => "l"]) |> CSV.write("./results/$(name)_$(censor_end)d_$(confidence)_params.csv")

    return admissible_parameters
end

SARS_α_space = 0.05:0.001:0.15
SARS_p_space = 0:0.01:1

calculate_admissible_parameters("SARS", 7, 90, SARS_α_space, SARS_p_space)
calculate_admissible_parameters("SARS", 14, 90, SARS_α_space, SARS_p_space)
calculate_admissible_parameters("SARS", 21, 90, SARS_α_space, SARS_p_space)
calculate_admissible_parameters("SARS", 117, 90, SARS_α_space, SARS_p_space)

# MERS

MERS_α_space = 0.20:0.01:0.40
MERS_p_space = 0:0.01:1

calculate_admissible_parameters("MERS_first", 7, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_first", 14, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_first", 21, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_first", 402, 90, MERS_α_space, MERS_p_space)

calculate_admissible_parameters("MERS_second", 7, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_second", 14, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_second", 21, 90, MERS_α_space, MERS_p_space)
calculate_admissible_parameters("MERS_second", 436, 90, MERS_α_space, MERS_p_space)

# Coronavirus

Corona_α_space = 0.01:0.001:0.1
Corona_p_space = 0:0.01:1

calculate_admissible_parameters("Coronavirus", 7, 20, Corona_α_space, Corona_p_space)
calculate_admissible_parameters("Coronavirus", 14, 20, Corona_α_space, Corona_p_space)
calculate_admissible_parameters("Coronavirus", 21, 20, Corona_α_space, Corona_p_space)
calculate_admissible_parameters("Coronavirus", 29, 20, Corona_α_space, Corona_p_space)
