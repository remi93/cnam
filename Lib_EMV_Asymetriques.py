# Loi Beta

# Beta Gauche

def Beta_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    # densité de Z (cas loi Beta)
    def densite_Z_Beta(z,a,b,LLambda,alpha,bbeta):
        from scipy.stats import beta

        c0=LLambda*(bbeta-alpha)
        c1=1/c0
        c2=(z-alpha)/c0
        c3=beta.pdf(c2, a,b, loc=0, scale=1)
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Beta_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_Beta(t,a,b,LLambda,numpy.min(data),BBeta))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Beta_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        EMV = scipy.optimize.minimize(Beta_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})

        print("grad",Le_Jacobien(EMV.x))
        print("EMV=",EMV.x)
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1

    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
    Log_V=-1.0*Beta_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_Beta(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()          # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    # résultats avec 4 (ou 6) chiffres après la virgule
    print("parametres gauche=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Beta_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_Beta(z,a,b,LLambda,min(data),BBeta) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_Beta(z,a,b,LLambda,min(data),BBeta)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_Beta_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Beta_G["50%"]="{:.6e}".format(FDR_Beta_G(50))
    Irg_Beta_G["55%"]="{:.6e}".format(FDR_Beta_G(55))
    Irg_Beta_G["60%"]="{:.6e}".format(FDR_Beta_G(60))
    Irg_Beta_G["65%"]="{:.6e}".format(FDR_Beta_G(65))
    Irg_Beta_G["70%"]="{:.6e}".format(FDR_Beta_G(70))
    Irg_Beta_G["75%"]="{:.6e}".format(FDR_Beta_G(75))
    Irg_Beta_G["80%"]="{:.6e}".format(FDR_Beta_G(80))
    
    
    return data_image,parametres , Irg_Beta_G


###############################################################
# Beta Droite

def Beta_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    # densité de w (cas loi Beta)
    def densite_W_Beta(w,a,b,LLambda,alpha,bbeta):
        from scipy.stats import beta

        c0=LLambda*(bbeta-alpha)
        c1=1/c0
        c2=(bbeta-w)/c0
        c3=beta.pdf(c2, a,b, loc=0, scale=1)
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Beta_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_W_Beta(t,a,b,LLambda,AAlpha,100))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Beta_D_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        EMV = scipy.optimize.minimize(Beta_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None),(0 , None),(1 , None) ],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
        #EMV = scipy.optimize.minimize(Beta_D_Log_Vraisemblance,X0 ,
                #method='Nelder-Mead',
                #options={'disp': None})
        
        print(EMV.x)
        print(Le_Jacobien(EMV.x)) # EMV.jac grdient
        print(EMV.success) # convergence de l'algo
        
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x  # estimateurs de a et b
    #LLambda=1
    
    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)
    Log_V=-1.0*Beta_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    #print("borne inf=",100-LLambda*(100-numpy.min(data)))
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_Beta(w,a,b,LLambda,AAlpha,100) for w in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Beta_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_Beta(w,a,b,LLambda,AAlpha,BBeta) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_Beta(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_Beta_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Beta_D["50%"]="{:.6e}".format(FDR_Beta_D(50))
    Irg_Beta_D["55%"]="{:.6e}".format(FDR_Beta_D(55))
    Irg_Beta_D["60%"]="{:.6e}".format(FDR_Beta_D(60))
    Irg_Beta_D["65%"]="{:.6e}".format(FDR_Beta_D(65))
    Irg_Beta_D["70%"]="{:.6e}".format(FDR_Beta_D(70))
    Irg_Beta_D["75%"]="{:.6e}".format(FDR_Beta_D(75))
    Irg_Beta_D["80%"]="{:.6e}".format(FDR_Beta_D(80))
    
    return data_image,parametres , Irg_Beta_D


####################################################################################
# Loi Gamma

# Gamma Gauche

def Gamma_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    # densité de Z (cas loi Gamma)
    def densite_Z_gamma(z,a,b,LLambda,alpha,beta):
        from scipy.stats import gamma

        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(z-alpha)/c0
        c3=gamma.pdf(c2, a, loc=0, scale=1/b)/gamma.cdf(1, a, loc=0, scale=1/b)
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Gamma_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_gamma(t,a,b,LLambda,numpy.min(data),BBeta))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Gamma_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        EMV = scipy.optimize.minimize(Gamma_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None),(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})

        # EMV.jac grdient
        print("grad",Le_Jacobien(EMV.x))
        print("EMV=",EMV.x)
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b=EMV.x  # estimateurs de a et b et lambda
    #LLambda=1
    #a,b,LLambda=EMV.x  # estimateurs de a et b et lambda

    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))

    
    Log_V=-1.0*Gamma_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_gamma(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel('$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    print("parametres gauche=",parametres)
    
        # résultats pour le taux d'irrégularité
    
    def FDR_Gamma_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_Gamma(z,a,b,LLambda,min(data),BGamma) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_Gamma(z,a,b,LLambda,min(data),BGamma)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_Gamma_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Gamma_G["50%"]="{:.6e}".format(FDR_Gamma_G(50))
    Irg_Gamma_G["55%"]="{:.6e}".format(FDR_Gamma_G(55))
    Irg_Gamma_G["60%"]="{:.6e}".format(FDR_Gamma_G(60))
    Irg_Gamma_G["65%"]="{:.6e}".format(FDR_Gamma_G(65))
    Irg_Gamma_G["70%"]="{:.6e}".format(FDR_Gamma_G(70))
    Irg_Gamma_G["75%"]="{:.6e}".format(FDR_Gamma_G(75))
    Irg_Gamma_G["80%"]="{:.6e}".format(FDR_Gamma_G(80))

    
    return data_image,parametres , Irg_Gamma_G

#################################################################################
# Gamma Droite

def Gamma_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    # densité de w  (cas loi Gamma)
    def densite_W_gamma(w,a,b,LLambda,alpha,beta):
        from scipy.stats import gamma

        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=gamma.pdf(c2, a, loc=0, scale=1/b)/gamma.cdf(1, a, loc=0, scale=1/b)
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Gamma_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_W_gamma(t,a,b,LLambda,AAlpha,100))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Gamma_D_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        EMV = scipy.optimize.minimize(Gamma_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(1e-07 , None),(1e-07 , None),(1 , None) ],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
        #EMV = scipy.optimize.minimize(Beta_D_Log_Vraisemblance,X0 ,
                #method='Nelder-Mead',
                #options={'disp': None})
        
        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
        print(EMV.success) # convergence de l'algo
        
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x  # estimateurs de a et b
    #LLambda=1

    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)

    Log_V=-1.0*Gamma_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    #print("borne inf=",100-LLambda*(100-numpy.min(data)))
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_gamma(w,a,b,LLambda,AAlpha,100) for w in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel('$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Gamma_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_Gamma(w,a,b,LLambda,AAlpha,BGamma) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_gamma(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_Gamma_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Gamma_D["50%"]="{:.6e}".format(FDR_Gamma_D(50))
    Irg_Gamma_D["55%"]="{:.6e}".format(FDR_Gamma_D(55))
    Irg_Gamma_D["60%"]="{:.6e}".format(FDR_Gamma_D(60))
    Irg_Gamma_D["65%"]="{:.6e}".format(FDR_Gamma_D(65))
    Irg_Gamma_D["70%"]="{:.6e}".format(FDR_Gamma_D(70))
    Irg_Gamma_D["75%"]="{:.6e}".format(FDR_Gamma_D(75))
    Irg_Gamma_D["80%"]="{:.6e}".format(FDR_Gamma_D(80))
    
    return data_image,parametres , Irg_Gamma_D


####################################################################################
# Loi de Weibull

## Weibull gauche

def Weibull_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    def densite_Weibull(x,a,b):
        import numpy as np
        return (a/b)*np.power(x/b,a-1)*np.exp(-np.power(x/b,a))
    
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
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Weibull_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_Weibull(t,a,b,LLambda,numpy.min(data),BBeta))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Weibull_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        EMV = scipy.optimize.minimize(Weibull_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})

        print("grad",Le_Jacobien(EMV.x))
        print("EMV=",EMV.x)
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1

    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
    Log_V=-1.0*Weibull_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_Weibull(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    print("parametres gauche=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Weibull_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_Weibull(z,a,b,LLambda,min(data),BWeibull) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_Weibull(z,a,b,LLambda,min(data),BWeibull)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_Weibull_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Weibull_G["50%"]="{:.6e}".format(FDR_Weibull_G(50))
    Irg_Weibull_G["55%"]="{:.6e}".format(FDR_Weibull_G(55))
    Irg_Weibull_G["60%"]="{:.6e}".format(FDR_Weibull_G(60))
    Irg_Weibull_G["65%"]="{:.6e}".format(FDR_Weibull_G(65))
    Irg_Weibull_G["70%"]="{:.6e}".format(FDR_Weibull_G(70))
    Irg_Weibull_G["75%"]="{:.6e}".format(FDR_Weibull_G(75))
    Irg_Weibull_G["80%"]="{:.6e}".format(FDR_Weibull_G(80))

    
    return data_image,parametres , Irg_Weibull_G 


########################################################"
## Weibull droite

def Weibull_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    def densite_Weibull(x,a,b):
        import numpy as np
        return np.divide(a,b)*np.power(np.divide(x,b),a-1)*np.exp(-np.power(np.divide(x,b),a))
    
    def fdr_Weibull(x,a,b):
        import numpy as np
        return 1.0-np.exp(-np.power(np.divide(x,b),a))
    
    # densité de w (cas loi Weibull)
    def densite_W_Weibull(w,a,b,LLambda,alpha,beta):

        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=densite_Weibull(c2,a,b)/fdr_Weibull(1,a,b)
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Weibull_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        #LLambda=1
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            A=numpy.log(densite_W_Weibull(t,a,b,LLambda,AAlpha,100))
            LL+=A
        #print(LL)
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Weibull_D_Log_Vraisemblance,x)

    #Lambda_sup=100/(100-numpy.min(data))
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        EMV = scipy.optimize.minimize(Weibull_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1 , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1
    #a,b,LLambda=EMV.x
    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)
    Log_V=-1.0*Weibull_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_Weibull(z,a,b,LLambda,AAlpha,100) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    

    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Weibull_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_Weibull(w,a,b,LLambda,AAlpha,BWeibull) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_Weibull(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_Weibull_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Weibull_D["50%"]="{:.6e}".format(FDR_Weibull_D(50))
    Irg_Weibull_D["55%"]="{:.6e}".format(FDR_Weibull_D(55))
    Irg_Weibull_D["60%"]="{:.6e}".format(FDR_Weibull_D(60))
    Irg_Weibull_D["65%"]="{:.6e}".format(FDR_Weibull_D(65))
    Irg_Weibull_D["70%"]="{:.6e}".format(FDR_Weibull_D(70))
    Irg_Weibull_D["75%"]="{:.6e}".format(FDR_Weibull_D(75))
    Irg_Weibull_D["80%"]="{:.6e}".format(FDR_Weibull_D(80))

    
    return  data_image,parametres, Irg_Weibull_D


####################################################################################
# Loi Normale

## Normale gauche

def Normale_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
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
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Normale_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        #ss=max(100,a+4*numpy.sqrt(b))
        #print("ss",ss)
        #LLambda=(ss-numpy.min(data))/(BBeta-numpy.min(data))
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_Normale(t,a,b,LLambda,numpy.min(data),BBeta))
        
        #print(LL)
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Normale_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[numpy.mean(data)*numpy.random.rand(),numpy.var(data)*numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        
        EMV = scipy.optimize.minimize(Normale_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(None , None) ,(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
##        EMV = scipy.optimize.minimize(Normale_G_Log_Vraisemblance,X0 ,
##                method='Nelder-Mead', bounds =[(0 , None) ,(1e-02 , None)] , tol=1e-8,
##                options={'disp': None})

        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1

    #a,b,LLambda=EMV.x
    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
    
    Log_V=-1.0*Normale_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_Normale(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel('$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    # résultats avec 4 (ou 6) chiffres après la virgule
    print("parametres gauche=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Normale_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_Normale(z,a,b,LLambda,min(data),BNormale) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_Normale(z,a,b,LLambda,min(data),BNormale)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_Normale_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Normale_G["50%"]="{:.6e}".format(FDR_Normale_G(50))
    Irg_Normale_G["55%"]="{:.6e}".format(FDR_Normale_G(55))
    Irg_Normale_G["60%"]="{:.6e}".format(FDR_Normale_G(60))
    Irg_Normale_G["65%"]="{:.6e}".format(FDR_Normale_G(65))
    Irg_Normale_G["70%"]="{:.6e}".format(FDR_Normale_G(70))
    Irg_Normale_G["75%"]="{:.6e}".format(FDR_Normale_G(75))
    Irg_Normale_G["80%"]="{:.6e}".format(FDR_Normale_G(80))
    
    return data_image,parametres , Irg_Normale_G

###########################################
## Normale droite

def Normale_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
    # densité de W (cas loi Normale)
    def densite_W_Normale(w,mu,sigma2,LLambda,alpha,beta):
        from scipy.stats import norm
        import numpy as np

        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0

        diffdr=norm.cdf(1,  loc=mu, scale=np.sqrt(sigma2))-norm.cdf(0,  loc=mu, scale=np.sqrt(sigma2))
        c3=norm.pdf(c2,  loc=mu, scale=np.sqrt(sigma2))/diffdr
        return c1*c3
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def Normale_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        #ss=min(numpy.min(data),a-4*numpy.sqrt(b))
        #print("ss",ss)
        #LLambda=(100-ss)/(100-AAlpha)
        
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            A=numpy.log(densite_W_Normale(t,a,b,LLambda,AAlpha,100))
            LL+=A
        #print(LL)
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(Normale_D_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[numpy.mean(data)*numpy.random.rand(),numpy.var(data)*numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        
        EMV = scipy.optimize.minimize(Normale_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(None , None) ,(0 , None),(1e-05 , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
##        EMV = scipy.optimize.minimize(Normale_D_Log_Vraisemblance,X0 ,
##                method='Nelder-Mead', bounds =[(0 , None) ,(1e-02 , None)] , tol=1e-8,
##                options={'disp': None})
        
        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-2:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1
    #a,b,LLambda=EMV.x
    
    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)
    Log_V=-1.0*Normale_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_Normale(z,a,b,LLambda,AAlpha,100) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel('$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_Normale_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_Normale(w,a,b,LLambda,AAlpha,BNormale) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_Normale(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_Normale_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_Normale_D["50%"]="{:.6e}".format(FDR_Normale_D(50))
    Irg_Normale_D["55%"]="{:.6e}".format(FDR_Normale_D(55))
    Irg_Normale_D["60%"]="{:.6e}".format(FDR_Normale_D(60))
    Irg_Normale_D["65%"]="{:.6e}".format(FDR_Normale_D(65))
    Irg_Normale_D["70%"]="{:.6e}".format(FDR_Normale_D(70))
    Irg_Normale_D["75%"]="{:.6e}".format(FDR_Normale_D(75))
    Irg_Normale_D["80%"]="{:.6e}".format(FDR_Normale_D(80))
    
    return data_image,parametres , Irg_Normale_D


####################################################################################
# Loi Covolution Exponentielle

## Covolution Exponentielle gauche

def ConvExp_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
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

    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def ConvExp_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_ConvExp(t,a,b,LLambda,numpy.min(data),BBeta))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(ConvExp_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        EMV = scipy.optimize.minimize(ConvExp_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})

        print("grad",Le_Jacobien(EMV.x))
        print("EMV=",EMV.x)
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1

    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
    Log_V=-1.0*ConvExp_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_ConvExp(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    print("parametres gauche=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_ConvExp_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_ConvExp(z,a,b,LLambda,min(data),BConvExp) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_ConvExp(z,a,b,LLambda,min(data),BConvExp)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_ConvExp_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_ConvExp_G["50%"]="{:.6e}".format(FDR_ConvExp_G(50))
    Irg_ConvExp_G["55%"]="{:.6e}".format(FDR_ConvExp_G(55))
    Irg_ConvExp_G["60%"]="{:.6e}".format(FDR_ConvExp_G(60))
    Irg_ConvExp_G["65%"]="{:.6e}".format(FDR_ConvExp_G(65))
    Irg_ConvExp_G["70%"]="{:.6e}".format(FDR_ConvExp_G(70))
    Irg_ConvExp_G["75%"]="{:.6e}".format(FDR_ConvExp_G(75))
    Irg_ConvExp_G["80%"]="{:.6e}".format(FDR_ConvExp_G(80))
    
    return data_image,parametres , Irg_ConvExp_G


########################################################"
## Covolution Exponentielle droite


def ConvExp_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt


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
    
    # densité de w (cas loi Covolution Exponentielle)

    def densite_W_ConvExp(w,a,b,LLambda,alpha,beta):
    
        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=densite_ConvExp(c2,a,b)/fdr_ConvExp(1,a,b)
        return c1*c3
            
    
    # densité de w (cas loi Covolution Exponentielle)

    def densite_W_ConvExp(w,a,b,LLambda,alpha,beta):
        
        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=densite_ConvExp(c2,a,b)/fdr_ConvExp(1,a,b)
        return c1*c3
    
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def ConvExp_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        #LLambda=1
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            A=numpy.log(densite_W_ConvExp(t,a,b,LLambda,AAlpha,100))
            LL+=A
        #print(LL)
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(ConvExp_D_Log_Vraisemblance,x)

    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        EMV = scipy.optimize.minimize(ConvExp_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1 , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1
    #a,b,LLambda=EMV.x
    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)
    Log_V=-1.0*ConvExp_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_ConvExp(z,a,b,LLambda,AAlpha,100) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()

    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_ConvExp_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_ConvExp(w,a,b,LLambda,AAlpha,BConvExp) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_ConvExp(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_ConvExp_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_ConvExp_D["50%"]="{:.6e}".format(FDR_ConvExp_D(50))
    Irg_ConvExp_D["55%"]="{:.6e}".format(FDR_ConvExp_D(55))
    Irg_ConvExp_D["60%"]="{:.6e}".format(FDR_ConvExp_D(60))
    Irg_ConvExp_D["65%"]="{:.6e}".format(FDR_ConvExp_D(65))
    Irg_ConvExp_D["70%"]="{:.6e}".format(FDR_ConvExp_D(70))
    Irg_ConvExp_D["75%"]="{:.6e}".format(FDR_ConvExp_D(75))
    Irg_ConvExp_D["80%"]="{:.6e}".format(FDR_ConvExp_D(80))
    
    return  data_image,parametres , Irg_ConvExp_D


####################################################################################
# Loi Hyper Exponentielle

## Hyper Exponentielle gauche

def HyperExp_Gauche_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt
    
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

    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def HyperExp_G_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        BBeta=param[2]
        #LLambda=param[2]
        #LLambda=1
        LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            LL+=numpy.log(densite_Z_HyperExp(t,a,b,LLambda,numpy.min(data),BBeta))
        
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(HyperExp_G_Log_Vraisemblance,x)
    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)+numpy.random.rand()]
        EMV = scipy.optimize.minimize(HyperExp_G_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1+numpy.min(data) , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})

        print("grad",Le_Jacobien(EMV.x))
        print("EMV=",EMV.x)
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1

    a,b,BBeta=EMV.x
    LLambda=(100-numpy.min(data))/(BBeta-numpy.min(data))
    Log_V=-1.0*HyperExp_G_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_Z_HyperExp(z,a,b,LLambda,numpy.min(data),BBeta) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$z$', fontsize=15)
    plt.ylabel(r'$f(z)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f , \hat{\lambda}=%.4f ,\hat{\beta}=%.4f ,\ell=%.6f$"%(a,b,LLambda,BBeta,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()
    
    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(BBeta, 4) , numpy.round(Log_V, 6)]

    print("parametres gauche=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_HyperExp_G(seuil):
        ### F(seuil)=int_{min(data)}^{seuil} densite_Z_HyperExp(z,a,b,LLambda,min(data),BHyperExp) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= min(data):
            return 0
        else:
            fonction= lambda z : densite_Z_HyperExp(z,a,b,LLambda,min(data),BHyperExp)
            return integrate.quad(fonction, min(data), seuil)[0]
        
    Irg_HyperExp_G={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_HyperExp_G["50%"]="{:.6e}".format(FDR_HyperExp_G(50))
    Irg_HyperExp_G["55%"]="{:.6e}".format(FDR_HyperExp_G(55))
    Irg_HyperExp_G["60%"]="{:.6e}".format(FDR_HyperExp_G(60))
    Irg_HyperExp_G["65%"]="{:.6e}".format(FDR_HyperExp_G(65))
    Irg_HyperExp_G["70%"]="{:.6e}".format(FDR_HyperExp_G(70))
    Irg_HyperExp_G["75%"]="{:.6e}".format(FDR_HyperExp_G(75))
    Irg_HyperExp_G["80%"]="{:.6e}".format(FDR_HyperExp_G(80))
    
    return data_image,parametres , Irg_HyperExp_G


########################################################"
## Hyper Exponentielle droite


def HyperExp_Droite_EMV64(data,nb_classe=10):
    
    import warnings
    warnings.filterwarnings('ignore')
    
    import numpy
    import scipy.optimize
    
    import base64
    import io
    import matplotlib.pyplot as plt

    
    def densite_HyperExp(x,a,b):
        import numpy as np
        c1=a*b/(a+b)
        return c1*(np.exp(-a*x)+np.exp(-b*x))
    
    def fdr_HyperExp(x,a,b):
        import numpy as np
        c1=a*b/(a+b)
        return 1-c1*(np.exp(-a*x)/a+np.exp(-b*x)/b)
    
    # densité de w (cas loi Hyper Exponentielle)

    def densite_W_HyperExp(w,a,b,LLambda,alpha,beta):
    
        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=densite_HyperExp(c2,a,b)/fdr_HyperExp(1,a,b)
        return c1*c3
            
    
    # densité de w (cas loi Covolution Exponentielle)

    def densite_W_HyperExp(w,a,b,LLambda,alpha,beta):
        
        c0=LLambda*(beta-alpha)
        c1=1/c0
        c2=(beta-w)/c0
        c3=densite_HyperExp(c2,a,b)/fdr_HyperExp(1,a,b)
        return c1*c3
    
    ################################################################
    # calcul (numérique) du vecteur gradien
    def Gradient(F,x,h=1.0e-6):
        def variation_en_k(F,x,k):
            y_G=[x[j]-0.5*h if j==k else x[j] for j in range(len(x))]
            y_D=[x[j]+0.5*h if j==k else x[j] for j in range(len(x))]
            return (F(y_D)-F(y_G))/(1.0*h)
        # G vecteur gradient
        G=[]
        for j in range(len(x)):
            G.append(variation_en_k(F,x, j))
        return G

    def HyperExp_D_Log_Vraisemblance(param):
        import numpy
        a=param[0]
        b=param[1]
        AAlpha=param[2]
        #LLambda=param[2]
        LLambda=(100-numpy.min(data))/(100-AAlpha)
        
        #LLambda=1
        LL=0
        for t in [u for u in data if u!=numpy.min(data)]:
            A=numpy.log(densite_W_HyperExp(t,a,b,LLambda,AAlpha,100))
            LL+=A
        #print(LL)
        return -1.0*LL  # -1 : on cherche le maximum
        
    def Le_Jacobien(x):
        return Gradient(HyperExp_D_Log_Vraisemblance,x)

    
    while True:
        #X0=numpy.random.rand(3)
        X0=[1+numpy.random.rand(),1+numpy.random.rand(),numpy.min(data)*numpy.random.rand()]
        EMV = scipy.optimize.minimize(HyperExp_D_Log_Vraisemblance,X0 ,
                method='SLSQP',jac=Le_Jacobien, bounds =[(0 , None) ,(0 , None),(1 , None)],
                options={'maxiter': 5500,'ftol': 1e-10,'disp': None})
        
        print("EMV=",EMV.x)
        print("grad=",Le_Jacobien(EMV.x))
    
        if EMV.success==True and numpy.max(numpy.abs(Le_Jacobien(EMV.x))) <1e-3:
            break

    # graphe de la densité estiméé
    #a,b,LLambda=EMV.x  # estimateurs de a et b
    #a,b=EMV.x
    #LLambda=1
    #a,b,LLambda=EMV.x
    a,b,AAlpha=EMV.x
    LLambda=(100-numpy.min(data))/(100-AAlpha)
    Log_V=-1.0*HyperExp_D_Log_Vraisemblance(EMV.x) # valeurv estimée de log_Vraisemblance
    
    # pour histogramme
    
    ### classes pour l'histogramme
    def classes_continues(data,nb_classe=5):
        import functools
        e0=functools.reduce(lambda x,y: x if x<= y else y , data) # min(data)
        ek=functools.reduce(lambda x,y: x if x>= y else y , data) # max(data)
        Extremites=[e0+(ek-e0)*i/(1.0*nb_classe) for i in range(nb_classe+1)]
        Compte=[len([i for i in range(len(data)) if data[i]>= Extremites[k] and data[i]<= Extremites[k+1]]) for k in range(nb_classe)]
        
        return Extremites, Compte
    
    ###############
    Extremites, Compte=classes_continues(data,nb_classe)
    Centre=[0.5*(Extremites[k]+Extremites[k+1])  for k in range(nb_classe)]
    ecart=Extremites[1]-Extremites[0]
    
    nb_point=200
    
    intervalle=[numpy.min(data)+(100-numpy.min(data))*i/(nb_point-1) for i in range(nb_point)]
    
    y=[densite_W_HyperExp(z,a,b,LLambda,AAlpha,100) for z in intervalle]
    
    # graphe
    ### Axes
    def Axes_simple1(axe):
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        axe.xaxis.set_ticks_position('bottom')
        axe.yaxis.set_ticks_position('left')
    
    fig, ax = plt.subplots(figsize=(10,7))
    Axes_simple1(ax)
    ax.set_facecolor([0.75,0.75,0.65])
    
    # graphe de la dinsité estimée
    plt.plot(intervalle,y,color=[0.25,0.25,1],lw=3)
    
    # normalisation courbe_histo
    alpha=numpy.max(y)/numpy.max(Compte)
    Compte1=[alpha*n for n in Compte]
    
    for k in range(nb_classe):
        plt.bar(Centre[k], Compte1[k], width=ecart,color=numpy.random.rand(3),ec="white")
    
    
    plt.xlabel(r'$w$', fontsize=15)
    plt.ylabel(r'$f(w)$', fontsize=15)
    plt.title(r"$\hat{a}=%.4f , \hat{b}=%.4f ,\hat{\lambda}=%.4f ,\hat{\alpha}=%.4f ,\ell=%.6f$"%(a,b,LLambda,AAlpha,Log_V), fontsize=13)
    
    # créer une image PNG en mémoire
    img = io.BytesIO()              # 
    plt.savefig(img, format='png')  # 
    img.seek(0)                     # 
    
    data_image = img.getvalue()         # récupérer les données du fichier (BytesIO)
    data_image = base64.b64encode(data_image) # convertir en base64 en octets
    data_image = data_image.decode()    # convertir des octets en chaîne
    
    plt.close()

    # résultats avec 4 (ou 6) chiffres après la virgule

    parametres=[numpy.round(a, 4) , numpy.round(b, 4), numpy.round(LLambda, 4), numpy.round(AAlpha, 4) , numpy.round(Log_V, 6)]

    print("parametres droite=",parametres)
    
    # résultats pour le taux d'irrégularité
    
    def FDR_HyperExp_D(seuil):
        ### F(seuil)=int_{100-LLambda(100-AAlpha)}^{seuil} densite_W_HyperExp(w,a,b,LLambda,AAlpha,BHyperExp) dz
        
        from scipy import integrate  # pour intégrale
        
        if seuil <= 100-LLambda*(100-AAlpha):
            return 0
        else:
            fonction= lambda w : densite_W_HyperExp(w,a,b,LLambda,AAlpha,100)
            return integrate.quad(fonction, 100-LLambda*(100-AAlpha), seuil)[0]
        
    Irg_HyperExp_D={}  #Dictionnaire pour les différents seuils d'irrégularité.
    Irg_HyperExp_D["50%"]="{:.6e}".format(FDR_HyperExp_D(50))
    Irg_HyperExp_D["55%"]="{:.6e}".format(FDR_HyperExp_D(55))
    Irg_HyperExp_D["60%"]="{:.6e}".format(FDR_HyperExp_D(60))
    Irg_HyperExp_D["65%"]="{:.6e}".format(FDR_HyperExp_D(65))
    Irg_HyperExp_D["70%"]="{:.6e}".format(FDR_HyperExp_D(70))
    Irg_HyperExp_D["75%"]="{:.6e}".format(FDR_HyperExp_D(75))
    Irg_HyperExp_D["80%"]="{:.6e}".format(FDR_HyperExp_D(80))
    
    return  data_image,parametres , Irg_HyperExp_D


####################################################################################
