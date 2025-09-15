using LinearAlgebra

include("barzilai_borwein.jl")

function gradient_wolfe_algorithm(x, a, m1,
    m2, simulator::Function, k_max, epsilon)
    start_time = time()
    f, g = simulator(x)
    # vec_f = [f]
    k = 0
    while norm(g) > epsilon && k < k_max
    # while norm(g) > epsilon     
        t = wolfe_search(x, -g, simulator, a, m1, m2)
        x -= t * g
        f, g = simulator(x)
        # push!(vec_f, f)
        k += 1
    end
    elapsed = time() - start_time
    optimal_value, _ = simulator(x)
    return x, elapsed, optimal_value, k
end