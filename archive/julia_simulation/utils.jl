using Distributed

# found here: https://discourse.julialang.org/t/everywhere-include-does-not-work-with-relative-file-paths/9598
function include_everywhere(filepath)
    fullpath = joinpath(@__DIR__, filepath)
    @sync for p in procs()
        @async remotecall_wait(include, p, fullpath)
    end
end
