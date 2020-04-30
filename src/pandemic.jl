module Pandemic
    using DataFrames
    using CSV

    export Outbreak, load_outbreak, censor, censorbyffx

    struct Outbreak
        region::String
        censoring::String
        epidemic_curve::Vector{Int64}
        fatality_curve::Vector{Int64}
        recovery_curve::Vector{Int64}
    end

    function load_outbreak(region::String, filepath::String) :: Outbreak
        fullpath = joinpath(@__DIR__, filepath)

        df = CSV.read(fullpath)

        return Outbreak(
            region,
            "None",
            convert(Vector{Int64}, df.Infected),
            convert(Vector{Int64}, df.Dead),
            convert(Vector{Int64}, df.Recovered),
        )
    end

    function censor(outbreak::Outbreak, T::Int64) :: Outbreak
        @assert 0 < T <= size(outbreak.epidemic_curve, 1)

        return Outbreak(
            outbreak.name,
            "$(T)d",
            outbreak.epidemic_curve[1:T],
            outbreak.fatality_curve[1:T],
            outbreak.recovery_curve[1:T],
        )
    end

    function censorbyffx(outbreak::Outbreak, X::Int64) :: Outbreak
        T = findfirst(x -> x > X, cumsum(outbreak.epidemic_curve))

        if isnothing(T)
            T = size(outbreak.epidemic_curve, 1)
            println("No censoring applied.")
        else
            println("Case #$X occured after $T days.")
        end

        return Outbreak(
            outbreak.name,
            "$(X)FFX",
            outbreak.epidemic_curve[1:T],
            outbreak.fatality_curve[1:T],
            outbreak.recovery_curve[1:T]
        )
    end
end  # module Pandemic
