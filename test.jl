

using LinearAlgebra
using Printf
using KrylovKit


include("ellip_center.jl")
include("conjugate_gradient.jl")
include("gradient_optimal_step.jl")
include("fast_gradient_nesterov.jl")
include("barzilai_borwein.jl")
include("gradient_wolfe_algorithm.jl")
include("bfgs_wolfe_algorithm.jl")

function test_methods(n::Int)
    # vec = 5 * rand(n, 1)
    println("Building data for n = $n")
    lambda = 1

    vec =rand(0:1.0, n)
    #vec[1]=1
    #vec[n]=50000.0
    #A = Diagonal(vec)
    A=vec*vec'+lambda*(1.0 * I)(n)
    (E1, V1, info1) = eigsolve(A, ones(Float64,n), 1, :SR)
    @show E1
    (E2, V2, info2) = eigsolve(A, ones(Float64,n), 1, :LR)
    @show E2 
    cond=E2[1]/E1[1]
    println("Condition number is $(@sprintf("%.16f", cond))")
    b = -A * ones(n)
    # b=zeros(n)
    epsilon = 10^(-8)
    a = 10
    m1 = 0.1
    m2 = 0.9
    L = norm(A, 2)
    simulator = (x::AbstractVector{Float64}) -> begin
        f = 0.5 * x' * A * x - b' * x
        g = A * x - b
        return f, g
    end
    k_max = 1000

    methods = []
    optimals = []
    elapseds = []
    iterations=[]

    x0_base = rand(n)
    x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = ellip_center!(A, b, epsilon, x0)
    push!(methods, "ellip_center")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("Ellip center: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = conjugate_gradient(A, b, x0, epsilon)
    push!(methods, "conjugate_gradient")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("conjugate gradient: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = gradient_optimal_step(A, b, x0, epsilon)
    push!(methods, "gradient_optimal_step")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("gradient optimal step: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
     x0 = copy(x0_base)
    _, elapsed, optimal_value, iter  = fast_gradient_nesterov(A, b, x0, epsilon, L,k_max)
    push!(methods, "fast_gradient_nesterov")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("fast gradient nesterov: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = barzilai_borwein(A, b, epsilon, x0, simulator, a, m1, m2, true)
    push!(methods, "barzilai_borwein_true")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("barzilai_borwein true: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = barzilai_borwein(A, b, epsilon, x0, simulator, a, m1, m2, false)
    push!(methods, "barzilai_borwein_false")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("barzilai_borwein false: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    #x0 = copy(x0_base)
    #_, elapsed, optimal_value, iter = gradient_wolfe_algorithm(x0, a, m1, m2, simulator, k_max, epsilon)
    #push!(methods, "gradient_wolfe_algorithm")
    #push!(optimals, optimal_value)
    #push!(elapseds, elapsed)
    #push!(iterations, iter)
    #println("gradient_wolfe_algorithm: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    # x0 = copy(x0_base)
    # _, elapsed, optimal_value = BFGS_wolfe_algorithm(x0, a, m1, m2, simulator, k_max, epsilon)
    # push!(methods, "BFGS_wolfe_algorithm")
    # push!(optimals, optimal_value)
    # push!(elapseds, elapsed)
    # println("BFGS_wolfe_algorithm: optimal_value = $optimal_value, elapsed = $elapsed")

    return methods, optimals, elapseds, iterations
end

function main()
    # N = [50_000, 70_000, 100_000, 150_000, 200_000, 250_000, 350_000, 500_000, 700_000, 850_000, 1_000_000]
    
    # N = [10,20,30,40,50,100,150,200,300,400,500,600,700]
     N = [500]
    open("results_dense.txt", "w") do f
        for n in N
            methods, optimals, elapseds, iterations = test_methods(n)
            write(f, "n = $n\n")
            write(f, "\tmethod\t\toptimal_value\t\tT (s)\t\tIterations\n")
            for i in eachindex(optimals)
                write(f, "\t$(methods[i])\t\t$(optimals[i])\t\t$(elapseds[i])\t\t$(iterations[i])\n")
            end
        end
    end
end

main()
