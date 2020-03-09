using DataFrames
using CSV
using Distributions

df = "./data/time_series/diff/SARS.csv" |> CSV.File |> DataFrame

function probability_patient_dies_within_k_periods(k, α, p)
    return α * (1 - (1 - p)^k)
end

function confidence_interval(confidence::Int)::Vector{Float64}
    confidence /= 100
    return [(1 - confidence) / 2, confidence + (1 - confidence) / 2]
end

function calculate_admissible_parameters(name, df, confidence)
    admissible_parameters = []

    α_space = 0.05:0.01:0.15
    p_space = 0.01:0.01:1

    censor_end = 14

    # E = epidemic curve
    E = df.Infected[1:censor_end]

    # y = real death count
    true_deaths = sum(df.Dead[1:censor_end])

    function build_poisson_binomial_distribution(E, α, p)
        # size of the probability vector is the total number of infected patients
        probability_vector_size = round(Int, sum(E))

        probability_vector = Array{Float64, 1}(undef, probability_vector_size)

        ptr = 1
        for (t, n) in enumerate(E)
            D_t = probability_patient_dies_within_k_periods(censor_end - t, α, p)

            for i in 1:n
                probability_vector[ptr] = D_t

                ptr += 1
            end
        end

        return PoissonBinomial(probability_vector)
    end

    for α in α_space
        for p in p_space
            println("(α=$α, p=$p) Building Poisson Binomial distribution...")
            poisson_binomial_distribution = build_poisson_binomial_distribution(E, α, p)
            println("(α=$α, p=$p) Created distribution with μ=$(mean(poisson_binomial_distribution)) and σ^2=$(var(poisson_binomial_distribution))")

            println(confidence_interval(confidence))

            low_deaths, high_deaths = quantile.(poisson_binomial_distribution, confidence_interval(confidence))

            if low_deaths <= true_deaths <= high_deaths
                println("(α=$α, p=$p) Parameters are admissible!")
                push!(admissible_parameters, (α, p))
            end
        end
    end

    println("Writing parameters to file.")

    admissible_parameters |> DataFrame |> rename([1 => "α", 2 => "p"]) |> CSV.write("./results/$name_$(censor_end)d_$(confidence)_params.csv")

    return admissible_parameters
end

calculate_admissible_parameters("SARS", df, 95)
