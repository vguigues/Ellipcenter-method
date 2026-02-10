using LinearAlgebra

include("barzilai_borwein.jl")

function BFGS_wolfe_algorithm(x::AbstractVector{Float64}, a::Real,
    m1::Real, m2::Real, simulator::Function,
    k_max::Int, epsilon::Real)
    start_time = time()
    f, g = simulator(x)
    vec_f = [f]
    k = 0
    W = (1.0 * I)(length(x))
    while norm(g) > epsilon && k < k_max
        t = wolfe_search(x, Diagonal(-W * g), simulator, a, m1, m2)
        xp = x - t * W * g
        fp, gp = simulator(xp)
        s = xp - x
        y = gp - g
        W -= ((s * y' * W + W * y * s') / (y' * s)) + (1 + (y' * W * y / (y' * s))) * (s * s' / (y' * s))
        x = xp
        f = fp
        push!(vec_f, f)
        g = gp
        k += 1
    end
    elapsed = time() - start_time
    optimal_value, _ = simulator(x)
    return x, elapsed, optimal_value
end