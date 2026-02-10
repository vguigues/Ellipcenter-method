using LinearAlgebra

function f(x, A, b)
    return 0.5 * x' * A * x - b' * x
end


function conjugate_gradient(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64}, x::AbstractVector{Float64}, epsilon::Real)
    r = A * x - b
    k = 0
    start_time = time()
    dp = zeros(eltype(x), size(x))
    while norm(r, 2) > epsilon
        if k == 0
            d = r
        else
            num = r' * A * dp
            denom = dp' * A * dp
            theta = -num / denom
            d = r + theta * dp
        end
        num = d' * r
        denom = d' * A * d
        rho = -num / denom
        x += rho * d
        r = A * x - b
        dp = d
        k += 1
    end
    elapsed = time() - start_time
    optimal_value = f(x, A, b)
    return x, elapsed, optimal_value,k
end