# densité de Z (cas loi Beta)
def densite_Z_Beta(z,a,b,LLambda,alpha,bbeta):
    from scipy.stats import beta
    
    c0=LLambda*(bbeta-alpha)
    c1=1/c0
    c2=(z-alpha)/c0
    c3=beta.pdf(c2, a,b, loc=0, scale=1)
    return c1*c3

# densité de w (cas loi Beta)
def densite_W_Beta(w,a,b,LLambda,alpha,bbeta):
    from scipy.stats import beta
    
    c0=LLambda*(bbeta-alpha)
    c1=1/c0
    c2=(bbeta-w)/c0
    c3=beta.pdf(c2, a,b, loc=0, scale=1)
    return c1*c3

# densité de Z (cas loi Gamma)
def densite_Z_gamma(z,a,b,LLambda,alpha,beta):
    from scipy.stats import gamma
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(z-alpha)/c0
    c3=gamma.pdf(c2, a, loc=0, scale=1/b)/gamma.cdf(1, a, loc=0, scale=1/b)
    return c1*c3

# densité de w  (cas loi Gamma)
def densite_W_gamma(w,a,b,LLambda,alpha,beta):
    from scipy.stats import gamma
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(beta-w)/c0
    c3=gamma.pdf(c2, a, loc=0, scale=1/b)/gamma.cdf(1, a, loc=0, scale=1/b)
    return c1*c3

#### Weibull

def densite_Weibull(x,a,b):
    import numpy as np
    return (a/b)*np.power(x/b,a-1)*np.exp(-np.power(x/b,a))
    #return (a/b)*np.exp((a-1)*np.log(x/b)-np.power(x/b,a))

def fdr_Weibull(x,a,b):
    import numpy as np
    return 1.0-np.exp(-np.power(x/b,a))

# densité de Z (cas loi Weibull)
def densite_Z_Weibull(z,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(z-alpha)/c0
    c3=densite_Weibull(c2,a,b)/fdr_Weibull(1,a,b)
    return c1*c3

# densité de w (cas loi Weibull)
def densite_W_Weibull(w,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(beta-w)/c0
    c3=densite_Weibull(c2,a,b)/fdr_Weibull(1,a,b)
    return c1*c3


## Normale

# densité de Z (cas loi Normale)
def densite_Z_Normale(z,mu,sigma2,LLambda,alpha,beta):
    from scipy.stats import norm
    import numpy as np
    
    #sigma=numpy.sqrt(sigma2)
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(z-alpha)*c1
    
    diffdr=norm.cdf(1,  loc=mu, scale=np.sqrt(sigma2))-norm.cdf(0,  loc=mu, scale=np.sqrt(sigma2))
    c3=norm.pdf(c2,  loc=mu, scale=np.sqrt(sigma2))/diffdr
    return c1*c3


# densité de W (cas loi Normale)
def densite_W_Normale(w,mu,sigma2,LLambda,alpha,beta):
    from scipy.stats import norm
    import numpy as np
    
    #sigma=numpy.sqrt(sigma2)
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(beta-w)/c0
    
    diffdr=norm.cdf(1,  loc=mu, scale=np.sqrt(sigma2))-norm.cdf(0,  loc=mu, scale=np.sqrt(sigma2))
    c3=norm.pdf(c2,  loc=mu, scale=np.sqrt(sigma2))/diffdr
    return c1*c3

# base64

def image64_Beta(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_Beta(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_Beta(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    #plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda$=%s, $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data

def image64_Gamma(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_gamma(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_gamma(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    #plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda=%s$ $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data

def image64_Gamma(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_gamma(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_gamma(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    #plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda=%s$ $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data



def image64_Weibull(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_Weibull(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_Weibull(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    #plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda=%s$ ,  $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data


def image64_Normale(mu,sigma2,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_Normale(w,mu,sigma2,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_Normale(z,mu,sigma2,LLambda,alpha,beta) for z in intervalle]
        
    #plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(mu,sigma2,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(mu,sigma2,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"$\mu=%s$ , $\sigma^{2}$=%s, $\lambda=%s$ $\alpha$=%s , $\beta$=%s"%(mu,sigma2,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data

# Loi Covolution Exponentielle

def densite_ConvExp(x,a,b):
    import numpy as np
    if a==b:
        return a**2*x*np.exp(-a*x)
    else:
        c1=a*b/(b-a)
        return c1*(np.exp(-a*x)-np.exp(-b*x))

def fdr_ConvExp(x,a,b):
    import numpy as np
    if a==b:
        return 1-(1+a*x)*np.exp(-a*x)
    else:
        c1=a*b/(b-a)
        return 1+c1*(np.exp(-b*x)/b-np.exp(-a*x)/a)

# densité de Z (cas loi Covolution Exponentielle)
def densite_Z_ConvExp(z,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(z-alpha)/c0
    c3=densite_ConvExp(c2,a,b)/fdr_ConvExp(1,a,b)
    return c1*c3

# densité de w (cas loi Covolution Exponentielle)
def densite_W_ConvExp(w,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(beta-w)/c0
    c3=densite_ConvExp(c2,a,b)/fdr_ConvExp(1,a,b)
    return c1*c3

def image64_ConvExp(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_ConvExp(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_ConvExp(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda=%s$ ,  $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data

# Loi Hyper Exponentielle

def densite_HyperExp(x,a,b):
    import numpy as np
    c1=a*b/(a+b)
    return c1*(np.exp(-a*x)+np.exp(-b*x))

def fdr_HyperExp(x,a,b):
    import numpy as np
    c1=a*b/(a+b)
    return 1-c1*(np.exp(-a*x)/a+np.exp(-b*x)/b)

# densité de Z (cas loi Hyper Exponentielle)
def densite_Z_HyperExp(z,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(z-alpha)/c0
    c3=densite_HyperExp(c2,a,b)/fdr_HyperExp(1,a,b)
    return c1*c3

# densité de w (cas loi Hyper Exponentielle)
def densite_W_HyperExp(w,a,b,LLambda,alpha,beta):
    
    c0=LLambda*(beta-alpha)
    c1=1/c0
    c2=(beta-w)/c0
    c3=densite_HyperExp(c2,a,b)/fdr_HyperExp(1,a,b)
    return c1*c3

def image64_HyperExp(a,b,LLambda,alpha,beta,asymetrie="droite"):
    
    import base64
    import io
    import matplotlib.pyplot as plt
    import numpy

    # générer l'image  par matplotlib en format base64 pour Flask 
    
    intervalle=numpy.linspace(alpha,beta,300)
    
    if asymetrie=="droite":
        y = [densite_W_HyperExp(w,a,b,LLambda,alpha,beta) for w in intervalle]
        
    if asymetrie=="gauche":
        y=[densite_Z_HyperExp(z,a,b,LLambda,alpha,beta) for z in intervalle]
        
    plt.figure(figsize=(10,7))
    ax = plt.gca()
    ax.set_facecolor([0.85,0.85,0.65])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    plt.plot(intervalle, y,lw=4)
    
    if asymetrie=="droite":
        plt.xlabel(r"w", fontsize=15)
        plt.ylabel(r"$f(w,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
    
    if asymetrie=="gauche":
        plt.xlabel(r"z", fontsize=15)
        plt.ylabel(r"$f(z,%s,%s,%s,%s,%s)$"%(a,b,LLambda,alpha,beta),fontsize=15)
        
    plt.title(r"a=%s , b=%s,$\lambda=%s$ ,  $\alpha$=%s , $\beta$=%s"%(a,b,LLambda,alpha,beta),fontsize=15)
    plt.grid()

    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data = base64.b64encode(data) # convertir en base64 en octets
    data = data.decode()          # convertir des octets en chaîne
    
    plt.close()

    return data




