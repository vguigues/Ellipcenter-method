using LinearAlgebra

function f(x, A, b)
    return 0.5 * x' * A * x - b' * x
end


function fast_gradient_nesterov(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64}, x::AbstractVector{Float64}, epsilon::Real, L::Real,k_max)
    start_time = time()
    y = x
    A_nest = 0
    k=0
    while norm(A * x - b, 2) > epsilon && k < k_max
        a = (1 + sqrt(1 + 4 * L * A_nest)) / (2 * L)
        n_A = A_nest + a
        xt = (A_nest * y + a * x) / n_A
        n_y = xt + (b - A * xt) / L
        x = (n_A / a) * n_y - (A_nest / a) * y
        y = n_y
        A_nest = n_A
        k+=1
    end
    elapsed = time() - start_time
    optimal_value = f(x, A, b)
    return x, elapsed, optimal_value,k
end