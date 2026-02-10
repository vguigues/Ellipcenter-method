using LinearAlgebra

function f(x, A, b)
    return 0.5 * x' * A * x - b' * x
end

function gradient_optimal_step(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64}, x::AbstractVector{Float64}, epsilon::Real)
    r = A * x - b
    start_time = time()
    k=0
    while norm(r) > epsilon
        t = (norm(r, 2)^2) / (r' * A * r)
        x -= t * r
        r = A * x - b
        k+=1
    end
    elapsed = time() - start_time
    optimal_value = f(x, A, b)
    return x, elapsed, optimal_value,k
end