using DataFrames
using CSV
using Distributions


function probability_patient_dies_within_k_periods(k, α, p)
    return α * (1 - (1 - p)^k)
end

function confidence_interval(confidence::Int)::Vector{Float64}
    confidence /= 100
    return [(1 - confidence) / 2, confidence + (1 - confidence) / 2]
end

function calculate_admissible_parameters(name, censor_end, confidence)
    df = "./data/time_series/diff/$name.csv" |> CSV.File |> DataFrame

    α_space = 0.01:0.01:0.1
    p_space = 0.01:0.01:1

    CI = confidence_interval(confidence)

    # E = epidemic curve
    E = df.Infected[1:censor_end]
    println(E)

    # y = real death count
    true_deaths = sum(df.Dead[1:censor_end])
    println(true_deaths)


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

    admissible_parameters = []
    for α in α_space
        for p in p_space
            println("[$name] (α=$α, p=$p) Building Poisson Binomial distribution...")
            poisson_binomial_distribution = build_poisson_binomial_distribution(E, α, p)
            println("[$name] (α=$α, p=$p) Poisson Binomial distribution: μ=$(mean(poisson_binomial_distribution)), σ^2=$(var(poisson_binomial_distribution))")

            low_deaths, high_deaths = quantile.(poisson_binomial_distribution, CI)

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
