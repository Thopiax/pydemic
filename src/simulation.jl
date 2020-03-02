using DataFrames
using CSV
using Statistics

df = "../data/time_series/diff/SARS.csv" |> CSV.File |> DataFrame

function resample(ts, rate)
    ans = []

    for t in 1:rate:length(ts) - rate + 1
        push!(ans, sum(ts[t:t + rate - 1]))
    end

    return ans
end

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
function simulate_case_with_small_markov_chain(α::Float64, β::Float64)
    outcome_day = 0
    while true
        prob = rand()
        if prob < α
            return "Dead", outcome_day
        elseif prob < α + β
            return "Survived", outcome_day
        end
        outcome_day += 1
    end
end

function simulate_death_curve(X, simulation, simulation_params)
    T = length(X)

    ans = zeros(T)

    for (t, X_t) in enumerate(X)
        for _ in 1:X_t
            outcome, outcome_day = simulation(simulation_params...)

            if outcome == "Dead"
                if t + outcome_day <= T
                    ans[t + outcome_day] += 1
                end
            end

            # if outcome_day >= 0 && t + case_result <= T
            #     death_curve[t + case_result] += 1
            # end
        end
    end

    return ans
end

function mse(curves, mean_curve)
    N = length(curves)
    return vec(1/N .* sum((curves .- mean_curve) .^ 2, dims=1))
end

X = df.Infected
y = df.Dead

simulate_death_curve_with_small_markov_chain = (X, α, β) -> simulate_death_curve(X, simulate_case_with_small_markov_chain, [α, β])

granularity = 0.01

α_space = granularity:granularity:1
β_space = granularity:granularity:1

nsims = 100

T = length(X)

valid = []

for α in α_space
    for β in β_space
        println("Checking α=$α, β=$β")
        simulated = Array{Float64, 2}(undef, T, nsims)

        for i in 1:nsims
            simulated[:, i] = simulate_death_curve_with_small_markov_chain(X, α, β)
        end

        mean_simulated = mean(simulated, dims=2)

        mses = mse(simulated, mean_simulated)
        lowest_mse, highest_mse = quantile(mses, [0.05, 0.95])

        true_mse = mse(y, mean_simulated)[1]

        println("[$lowest_mse, $highest_mse] => $true_mse")

        if lowest_mse <= true_mse <= highest_mse
            println("Valid parameters. MSE=$true_mse")
            push!(valid_parameters, (α, β))
        end
    end
end
