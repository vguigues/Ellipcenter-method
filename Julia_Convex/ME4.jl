
using LinearAlgebra

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
        elseif (qt <= (q0 + m1 * qp0 * t) && (qpt < m2 * qp0))
            t_L = t
        else
            t_R = t
        end

        if (t_R == 10^50)
            t *= a
        else
            t = (t_L + t_R) / 2
        end
        if ((t_R-t_L)<=0.001)
            f_in = true
            continue
        end
    end
    return t
end


function ME4(x,epsilon,valuef::Function,gradientf::Function)
n=length(x)
gradient=gradientf(x)
fvalue=valuef(x) 
iter=1

start_time = time()
while ((norm(gradient)>epsilon)&&(iter<=1000))
#while ((abs(fvalue)>epsilon)&&(iter<=1000))    
      #Compute by dichotomy y such that f(y)=f(x-t \nabla f(x))=f(x)
      tL=0;
      tR=2;
      fL=fvalue
      fR=valuef(x+tR*gradient)
      while (fR<fvalue)
             tR=2*tR;
             fR=valuef(x+tR*gradient)
      end
      t=(tL+tR)/2;
      f=valuef(x+t*gradient)
      
      contd=1
      if (abs(f-fvalue)<=0.1)
         contd=0
      else
         contd=1
      end
      while contd==1
            if (f<fvalue)
                tL=t 
                fL=f
            else
                tR=t 
                fR=f 
            end
            t=(tL+tR)/2;
            f=valuef(x+t*gradient)
            if (abs(f-fvalue)<=0.1)
               contd=0
            else
               contd= 1
            end
      end
      
      y=x+t*gradient
      fvaluey=valuef(y)
      gradienty=gradientf(y)
      scalar=0
      for i=1:n
         scalar+=gradient[i]*gradienty[i]
      end
      scalar=scalar/(norm(gradient)*norm(gradienty))
      if ((scalar==1)||(scalar==-1))
       x=(x+y)/2
      else
        gradientyn=gradienty/norm(gradienty)
        ek=(x-y)/norm(x-y) 
        ctheta=0
        for i in 1:n
           ctheta-=ek[i]*gradientyn[i]
        end
        stheta=sqrt(abs(1-ctheta^2))
        constantterm=0
        for i=1:n
            constantterm-=ek[i]*gradienty[i]
        end
        w=gradienty+constantterm*ek
        d=w/norm(w)-(stheta/(2*ctheta))*ek
        p=0.5*(x+y)
        gp=gradientf(p)
        dec=0
        for i in 1:n
            dec+=d[i]*gp[i] 
        end
        if (dec>=0)
           x=p
           gradient=gradientf(x)    
        else 
           k = 0
           tp = 0.0
           tc=0.0
           while (abs(dec) > epsilon)
                 if k == 0
                     step = wolfe_search(p,d, simulatorf1, a, m1, m2)
                    # step=0.001
                 else
                    s = tc - tp
                    y = -dep+dec
                    if option
                       num = s* y
                       denom = y* y
                       step = num / denom
                    else
                       num = s* s
                       denom = s* y
                       step = num / denom
                    end
                 end
                 tp = tc
                 dep = dec
                 tc=tp-step*dep
                 dec=0
                 gc=gradientf(p+tc*d)
                 for i in 1:n
                     dec+=d[i]*gc[i] 
                 end
                 k += 1
           end
           x=p+tc*d
           fvalue=valuef(x)
           gradient=gradientf(x)
        end
      end
      iter+=1 
end
elapsed = time() - start_time
optimal_value = valuef(x)
return x, elapsed, optimal_value, iter
end
