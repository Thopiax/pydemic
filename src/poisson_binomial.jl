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

function confidence_interval(confidence::Int)::Vector{Float64}
    confidence /= 100
    return [(1 - confidence) / 2, confidence + (1 - confidence) / 2]
end

function calculate_admissible_parameters(name, censor_end, confidence)
    df = "./data/time_series/diff/$name.csv" |> CSV.File |> DataFrame

    α_space = 0.01:0.01:0.1
    p_space = 0.05:0.05:1

    CI = confidence_interval(confidence)

    # E = epidemic curve
    E = df.Infected[1:censor_end]

    # y = real death count
    true_deaths = sum(df.Dead[1:censor_end])

    admissible_parameters = []
    for α in α_space
        for p in p_space
            total_fatality_samples = simulate_total_fatality_samples(E, 10000, α, p)
            println("[$name] (α=$α, p=$p) Total Fatality samples: μ=$(mean(total_fatality_samples)), σ^2=$(var(total_fatality_samples))")

            low_deaths, high_deaths = quantile(total_fatality_samples, [0.05, 0.95])

            if low_deaths <= true_deaths <= high_deaths
                println("[$name] (α=$α, p=$p) Parameters are admissible! [$low_deaths, $high_deaths]")
                push!(admissible_parameters, (α, p))
            end
        end
    end

    println("[$name] Writing parameters to file.")

    rename(DataFrame(admissible_parameters), [1 => "α", 2 => "p"]) |> CSV.write("./results/$(name)_$(censor_end)d_$(confidence)_params.csv")

    return admissible_parameters
end

# calculate_admissible_parameters("SARS", 7, 90)
# calculate_admissible_parameters("SARS", 14, 90)
# calculate_admissible_parameters("SARS", 21, 90)

calculate_admissible_parameters("Coronavirus", 7, 90)
calculate_admissible_parameters("Coronavirus", 14, 90)
calculate_admissible_parameters("Coronavirus", 21, 90)
