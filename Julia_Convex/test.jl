
using LinearAlgebra
using Printf
using KrylovKit


include("ME.jl")
include("ME2.jl")
include("ME3.jl")
include("ME4.jl")
include("fast_gradient_nesterov.jl")
include("barzilai_borwein.jl")
include("ellip_center.jl")

function test_methods(n::Int)
    println("Building data for n = $n")
    
    # lambda = 1

     # vec = 5 * rand(n, 1)
    
     vec =rand(0:1.0, n)
     vec[1]=1
     vec[n]=40000.0
     A = Diagonal(vec)
     b =-A*ones(n)
    
    # lambda = 1
     
    #  alpha=zeros(n)
    #  beta=zeros(n)
    #  for i in 1:n
    #      alpha[i]=0.0001
    #  end
    #  for i in 1:n-1
    #      beta[i]=0.1
    #  end
    #  beta[n]=n

    #  gamma=0.0001

    #  vec =rand(n)
    #  vec[1]=1.0
    #  vec[n]=10
    # A = Diagonal(vec)
    # A=vec*vec'+lambda*(1.0 * I)(n)
    # (E1, V1, info1) = eigsolve(A, ones(Float64,n), 1, :SR)
    # @show E1
    # (E2, V2, info2) = eigsolve(A, ones(Float64,n), 1, :LR)
    # @show E2 
    # cond=E2[1]/E1[1]
    # println("Condition number is $(@sprintf("%.16f", cond))")
    epsilon = 0.01
    a = 10
    m1 = 0.1
    m2 = 0.9
    
    function f1(x::AbstractVector)
        n=length(x)
        fvalue = 0.5 * x' * A * x - b' * x
        #  g = A * x - b
     
        #  fvalue=x'*Diagonal(beta)*x
        #  aux=0
        #  for i=1:n
        #      aux+=exp(alpha[i]*x[i]^2)
        #  end
        #  return fvalue+log(aux)
         return fvalue
    end

    function f2(x::AbstractVector)
          fvalue=x'*Diagonal(beta)*x+exp(gamma*x'*A*x)
          return fvalue
    end

    function gradientf1(x::AbstractVector)
           n=length(x)
        #    S=0
        #    v=zeros(n)
        #    for i=1:n
        #        S+=exp(alpha[i]*x[i]^2)
        #    end
        #     for i=1:n
        #        v[i]=-2*beta[i]*x[i]-(2*alpha[i]*x[i]*exp(alpha[i]*x[i]^2))/S
        #     end
        #     for i=1:n
        #        v[i]=-2*beta[i]*x[i]
        #     end
            v = -A * x + b
           return v
    end


    function gradientf2(x::AbstractVector)
            n=length(x)
            return -2*gamma*exp(gamma*x'*A*x)*A*x-2*Diagonal(beta)*x
    end

    function simulatorf1(x::AbstractVector)
             n=length(x)
            #  fvalue=x'*Diagonal(beta)*x
             fvalue=0.5 * x' * A * x - b' * x
            #   aux=0
            #   for i=1:n
            #   aux+=exp(alpha[i]*x[i]^2)
            #   end
            #  v=zeros(n)
            #   for i=1:n
            #       v[i]=-2*beta[i]*x[i]-(2*alpha[i]*x[i]*exp(alpha[i]*x[i]^2))/aux
            #   end
            #   for i=1:n
            #       v[i]=-2*beta[i]*x[i]
            #   end
            # f=fvalue+log(aux)
            f=fvalue
            v=b-A*x
            return f,v
    end 


    function simulatorf2(x::AbstractVector)
             fvalue=x'*Diagonal(beta)*x+exp(gamma*x'*A*x)
             g=-2*gamma*exp(gamma*x'*A*x)*A*x-2*Diagonal(beta)*x
             return fvalue,g
    end

    methods = []
    optimals = []
    elapseds = []
    iterations=[]

    kmax=1000

     x0_base = rand(n)
     x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = ME4(x0,epsilon,f1,gradientf1)
    push!(methods, "ellip_center")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("Ellip center: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed)), iter=$iter")
    
    x0_base = rand(n)
     x0 = copy(x0_base)
    _, elapsed, optimal_value, iter = ME2(x0,epsilon,f1,gradientf1)
    push!(methods, "ellip_center 2")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("Ellip center 2: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed)), iter=$iter")
    

    #x0 = copy(x0_base)
    #_, elapsed, optimal_value, iter  = fast_gradient_nesterov(A, b, x0, epsilon, L,k_max)
    #push!(methods, "fast_gradient_nesterov")
    #push!(optimals, optimal_value)
    #push!(elapseds, elapsed)
    #push!(iterations, iter)
    #println("fast gradient nesterov: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed))")
    
    x0 = copy(x0_base)
    x, elapsed, optimal_value, iter = barzilai_borwein(epsilon, x0, simulatorf1, a, m1, m2, true)
    push!(methods, "barzilai_borwein_true")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("barzilai_borwein true: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed)), iter=$iter")

    x0 = copy(x0_base)
    x, elapsed, optimal_value, iter = barzilai_borwein(epsilon, x0, simulatorf1, a, m1, m2, false)
    push!(methods, "barzilai_borwein_false")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("barzilai_borwein false: optimal_value = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed)), iter=$iter")
    
    x0 = copy(x0_base)
    x, elapsed, optimal_value, k= ellip_center!(A,b,epsilon,x0)
    push!(methods, "ellipcenter 3")
    push!(optimals, optimal_value)
    push!(elapseds, elapsed)
    push!(iterations, iter)
    println("ellipcenter  3 = $optimal_value, elapsed = $(@sprintf("%.16f", elapsed)), iter=$iter")
   


    return methods, optimals, elapseds, iterations
end

function main()
    N = [10000]
    #N = [10]
    open("results_dense.txt", "w") do f
        for n in N
            println("$n")
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
