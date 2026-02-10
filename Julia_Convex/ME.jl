
function ME(x,epsilon,valuef::Function,gradientf::Function)
n=length(x)
gradient=gradientf(x) 
iter=1

start_time = time()
while ((norm(gradient)>epsilon)&&(iter<=1000))
      #Compute by dichotomy y such that f(y)=f(x-t \nabla f(x))=f(x)
      tL=0;
      tR=2;
      fvalue=valuef(x)
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
        stheta=sqrt(1-ctheta^2)
        constantterm=0
        for i=1:n
            constantterm-=ek[i]*gradienty[i]
        end
        w=gradienty+constantterm*ek
        d=w/norm(w)-(stheta/(2*ctheta))*ek
        t=100
        p=0.5*(x+y)+t*d
        f=valuef(p)
        while ((f>fvalue)&&(t>0.0001))
            t=t/2
            p=0.5*(x+y)+t*d
            f=valuef(p)
        end
        x=p;
        gradient=gradientf(x)
      end
      iter+=1 
    end
    elapsed = time() - start_time
    optimal_value = valuef(x)
    return x, elapsed, optimal_value, iter
  end