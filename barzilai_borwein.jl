using LinearAlgebra

function f(x, A, b)
    return 0.5 * x' * A * x - b' * x
end

function wolfe_search(x::AbstractVector{Float64}, d::AbstractVector{Float64},
    simulator::Function, a::Real, m1::Real, m2::Real)
    t = 1
    t_L = 0
    t_R = 10^50
    f_in = false
    while !f_in
        f1, g1 = simulator(x + t * d)
        f2, g2 = simulator(x)
        qt = f1
        qpt = d' * g1
        q0 = f2
        qp0 = d' * g2
        if (qt <= (q0 + m1 * qp0 * t)) && (qpt >= m2 * qp0)
            f_in = true
            continue
        elseif qt <= (q0 + m1 * qp0 * t) && (qpt < m2 * qp0)
            t_L = t
        else
            t_R = t
        end

        if t_R == 10^50
            t *= a
        else
            t = (t_L + t_R) / 2
        end
    end
    return t
end



function barzilai_borwein(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
    epsilon::Real,
    x::AbstractVector{Float64} , simulator::Function,
    a::Real, m1::Real, m2::Real, option::Bool)

    start_time = time()
    k = 0
    d = b - A * x
    xp = zeros(eltype(x), size(x))
    dp = zeros(eltype(d), size(d))
    while (norm(d, 2) > epsilon)
        if k == 0
            step = wolfe_search(x, d, simulator, a, m1, m2)
        else
            s = x - xp
            y = dp - d

            if option
                num = s' * y
                denom = y' * y
                step = num / denom
            else
                num = s' * s
                denom = s' * y
                step = num / denom
            end
        end
        xp = x
        dp = d
        x += step * d
        d = b - A * x
        k += 1
    end
    elapsed = time() - start_time
    optimal_value = f(x, A, b)
    return x, elapsed, optimal_value,k
end