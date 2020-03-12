using DataFrames
using CSV
using Statistics

function resample(ts, rate)
    ans = []

    for t in 1:rate:length(ts) - rate + 1
        push!(ans, sum(ts[t:t + rate - 1]))
    end

    return ans
end

@enum Outcome survived=1 dead=2

#=
Simulates a case of infection given a K, α and R.

This uses the long markov chain model with K+1+R states.

Returns -1 if patient survives.
Returns k > 0 representing the death reporting lag
=#
function simulate_case_with_reporting_delay(K::Int64, α::Float64, R::Int64)
    outcome_delay = -1

    for k in 0:K
        if rand() < α
            outcome_delay = k
            break
        end
    end

    if outcome_delay >= 0
        reporting_delay = rand(0:R)

        return outcome_delay + reporting_delay
    end

    return outcome_delay
end

#=
Simulate a case with the new model of a 3 state markov chain given an α and β.

Returns a status and the
=#
function simulate_infection(α::Float64, β::Float64)
    outcome_day = 0
    while true
        prob = rand()
        if prob < α
            return dead, outcome_day
        elseif prob < α + β
            return survived, outcome_day
        end
        outcome_day += 1
    end
end

function simulate_outbreak(X, simulation, simulation_params)
    T = length(X)
    ans = zeros(T)

    for (t, X_t) in enumerate(X)
        for _ in 1:X_t
            outcome, outcome_day = simulation(simulation_params...)

            if outcome == dead
                if t + outcome_day <= T
                    ans[t + outcome_day] += 1
                end
            end
        end
    end

    return ans
end

simulate_outbreak_new = (X, α, β) -> simulate_outbreak(X, simulate_infection, [α, β])

function mortality_rate(X, d)
    return sum(d) / sum(X)
end

function mse(sample_paths, mean_sample_path)
    N = size(sample_paths, 1)
    return vec(1/N .* sum((sample_paths .- mean_sample_path) .^ 2, dims=1))
end

function optimize_parameters(X, d, α_space, β_space, nsims)
    admissible_configurations = []
    mortality_rates = []

    y = mortality_rate(X, d)
    T = length(X)

    for α in α_space
        for β in β_space
            println("Checking α=$α, β=$β")

            sample_paths = Array{Float64, 2}(undef, T, nsims)
            sample_mortality_rates = Array{Float64, 1}(undef, nsims)

            for i in 1:nsims
                sample_paths[:, i] = simulate_outbreak_new(X, α, β)
                sample_mortality_rates[i] = mortality_rate(X, sample_paths[:, i])
            end

            mean_sample_path = mean(sample_paths, dims=2)
            sample_mses = mse(sample_paths, mean_sample_path)

            low_mse, high_mse = quantile(sample_mses, [0.05, 0.95])
            true_mse = mse(d, mean_sample_path)[1]

            println("[$low_mse, $high_mse] => $true_mse")

            println(quantile(sample_mortality_rates, [0.25, 0.5, 0.75]))

            if low_mse <= true_mse <= high_mse
                push!(admissible_configurations, (α, β))

                push!(mortality_rates, sample_mortality_rates...)

                println("Valid parameters: MSE=$true_mse")
                println("Mortality Rate: $(mean(sample_mortality_rates)) +- $(std(sample_mortality_rates))")
            end
        end
    end

    return admissible_configurations, mortality_rates
end

df = "./data/time_series/diff/SARS.csv" |> CSV.File |> DataFrame

X = resample(df.Infected[1:15], 2)
d = resample(df.Dead[1:15], 2)

α_space = 0.01:0.01:0.2
β_space = 0.05:0.05:0.8

nsims = 100

admissible_configurations, mortality_rates = optimize_parameters(X, d, α_space, β_space, nsims)
