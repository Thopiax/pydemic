module OutbreakData
    using DataFrames
    using CSV

    export OutbreakTimeline, load, censor, censorbyffx

    struct OutbreakTimeline
        name::String
        configname::String
        epidemic_curve::Array{Int64}
        fatality_curve::Array{Int64}
    end

    function load(name::String, filepath::String) :: OutbreakTimeline
        fullpath = joinpath(@__DIR__, filepath)

        df = CSV.read(fullpath)

        return OutbreakTimeline(
            name,
            "full",
            convert(Vector{Int64}, df.Infected),
            convert(Vector{Int64}, df.Dead)
        )
    end

    function censor(timeline::OutbreakTimeline, T::Int64)
        @assert 0 < T <= size(timeline.epidemic_curve, 1)

        return OutbreakTimeline(
            timeline.name,
            "$(T)d",
            timeline.epidemic_curve[1:T],
            timeline.fatality_curve[1:T]
        )
    end

    function censorbyffx(timeline::OutbreakTimeline, X::Int64)
        T = findfirst(x -> x > X, cumsum(timeline.epidemic_curve))

        if isnothing(T)
            T = size(timeline.epidemic_curve, 1)
            println("No censoring applied.")
        else
            println("Case #$X occured after $T days.")
        end

        return OutbreakTimeline(
            timeline.name,
            "$(X)FFX",
            timeline.epidemic_curve[1:T],
            timeline.fatality_curve[1:T]
        )
    end
end  # module OutbreakData
