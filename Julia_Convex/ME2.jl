function ME2(x,epsilon,valuef::Function,gradientf::Function)

n=length(x)
gradient=gradientf(x) 
fvalue=valuef(x)
iter=1

start_time = time()
while ((norm(gradient)>epsilon)&&(iter<=1000))
#while ((abs(fvalue)>epsilon)&&(iter<=1000))
   #println(iter)
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
      x=(x+y)/2 
      fvalue=valuef(x)
      gradient=gradientf(x)
      iter+=1
    end
    elapsed = time() - start_time
    optimal_value = valuef(x)
    return x, elapsed, optimal_value, iter
   end
