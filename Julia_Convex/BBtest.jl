


  p=0.5*(x+y)
        gp=gradientf(p)
        fp=valuef(p)
        dep=0
        for i in 1:n
            dep+=d[i]*gp[i] 
        end
        if (dep>=0)
           x=p;
           gradient=gradientf(x)    
        else 
           k = 0
           tp = 0.0
           dp = 0.0
           while (abs(dep) > epsilon)
                 if k == 0
                    step = wolfe_search(p,d, simulator, a, m1, m2)
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
                 tp = tc
                 dp = dc
                 x += step * d
                 f, d = simulator(x)
                 dep=0
                 for i in 1:n
                     dep+=d[i]*gp[i] 
                 end        
                 k += 1
           end
        end
      