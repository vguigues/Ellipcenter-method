using LinearAlgebra

function f(x, A, b)
    return 0.5 * x' * A * x - b' * x
end

function ellip_center!(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64}, epsilon::Real,
    x::AbstractVector{Float64})

    d = b - A * x
    start_time = time()
    k=0
    while norm(d, 2) > epsilon
        step = 2 * (d' * d) / (d' * A * d)
        y = x + step * d
        da = b - A * y
        if (norm(da,2)>0)
        step = 2 * (da' * da) / (da' * A * da)
        z = y + step * da
        v = z - y
        u = y - x
        aux=A*v
        aux1 = v' * aux
        aux2 = u' * A * u
        aux3 = u' * aux
        aux4 = u' * d
        aux5 = v' * d
        num = aux3 * aux5 - aux1 * aux4
        denom = (aux3^2) - aux1 * aux2
        if (denom != 0)
        alpha = num / denom
        num = aux3 * aux4 - aux2 * aux5
        beta = num / denom
        x .= (1 - alpha) * x + (alpha - beta) * y + beta * z
        else
        x=(x+y)/2     
        end
        else
        x=y    
        end
        d .= b - A * x
        k+=1
    end
    elapsed = time() - start_time
    optimal_value = f(x, A, b)
    return x, elapsed, optimal_value, k
end

