def Moments_r(data,r):
    n=len(data)
    S=[x**r for x in data]
    return sum(S)/n

def Moments_Centre_r(data,r):
    n=len(data)
    m=Moments_r(data,1)
    S=[(x-m)**r for x in data]
    return sum(S)/n

def Moments_DG_r(modalites,effectifs,r):
    n=sum([ni for ni in effectifs])
    S=[nn*x**r for nn,x in zip(effectifs,modalites)]
    return sum(S)/n

def Moments_Centre_DG_r(modalites,effectifs,r):
    n=sum([ni for ni in effectifs])
    m=Moments_DG_r(modalites,effectifs,1)
    S=[nn*(x-m)**r for  nn,x in zip(effectifs,modalites)]
    return sum(S)/n

def Moments_CG_r(Ext_G,Ext_D,effectifs,r):
    n=sum([ni for ni in effectifs])
    centre=[0.5*(x+y) for x,y in zip(Ext_G,Ext_D)]
    S=[nn*c**r for nn,c in zip(effectifs,centre)]
    return sum(S)/n

def Moments_Centre_CG_r(Ext_G,Ext_D,effectifs,r):
    n=sum([ni for ni in effectifs])
    centre=[0.5*(x+y) for x,y in zip(Ext_G,Ext_D)]
    m=Moments_CG_r(Ext_G,Ext_D,effectifs,1)
    S=[nn*(c-m)**r for  nn,c in zip(effectifs,centre)]
    return sum(S)/n

def modalites(data):
    import numpy
    modalite,compte = numpy.unique(data, return_counts=True)
    return modalite,compte

def notre_modalite(data):
    dict_M={}
    for x in sorted(data):
        dict_M[x]=data.count(x)
    return dict_M

def intervalle_compte(data,nb_intervalle):
    m0=min(data)
    m1=max(data)
    # pour k intervalles, il y a k+1 extrémités
    extremites=[m0+(m1-m0)*i/nb_intervalle for i in range(nb_intervalle+1)]
    
    comptes=[]
    
    for k in range(nb_intervalle):
        comptes.append(len([i for i in range(len(data)) if data[i]>=extremites[k] and data[i]<=extremites[k+1]]))
        
    return extremites, comptes


def Histo_Discrete(data,nom=None):
    
    import numpy
    import matplotlib.pyplot as plt
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
    
    # sous fonction pour compter les occurrences
    def comptage(data):
        data=sorted(data)
        Dic_compt={}
        for valeur in data:
            Dic_compt[valeur]=data.count(valeur)
        return Dic_compt

    D=comptage(data)
    
    valeurs=[k for k in D.keys()]
    effectifs=[v for v in D.values()]
    i_mode=numpy.argmax(effectifs)
    ### multi_mode
    indice_mode=[i for i in range(len(effectifs)) if effectifs[i]==effectifs[i_mode]]

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    
    ax1.set_xticks(valeurs)  ## positions des extrémités des classes
    ax1.set_xticklabels(valeurs ,fontsize=10, rotation=25)
    ax1.set_xlabel("Valeurs",fontsize=14)
    ax1.set_ylabel("Effectifes",fontsize=14)
    
    for k in range(len(valeurs)):
        if k not in indice_mode:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white")
        else:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",
                    color = [0.15,0.15,0.85],hatch="X", lw=1., zorder = 0)
        for i in range(len(valeurs)):
            ax1.text(valeurs[i], effectifs[i], "%d"%(effectifs[i]),fontsize=9,
                     horizontalalignment='center',verticalalignment='bottom',style='italic')
    
    if nom is None:
        plt.show()
    else:
        nom_fig=str(nom)+".png"
        plt.savefig(nom_fig, format="png")
        plt.close()

def DDG_vers_DDNG(Modalites,Effectifs):
    Data=[]
    for v,N in zip(Modalites,Effectifs):
        Data.extend([v]*N)
    return Data

def Histo_Continue(data,k,nom=None):
    # k=nombre d'intervalles
    
    import numpy
    import matplotlib.pyplot as plt
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
    
    data=numpy.array([x for x in data])
    Ext=[min(data)+(max(data)-min(data))*i/(1.0*k) for i in range(k+1)]
    C=[0.5*(Ext[i]+Ext[i+1]) for i in range(k)]

    NN=[] # Effectifs des classes
    for i in range(k):
        NN.append(((Ext[i] <= data) & (data<=Ext[i+1])).sum())
        
    # pour la classe modale
    indice_max=[i for i in range(k) if NN[i]==numpy.max(NN)]
    
    TT=[str("{:.4f}".format(t)) for t in Ext]  ## pour afficher les extrémités des classes
    
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    largeur=Ext[1]-Ext[0]  #  largeur des classes
    
    for i in range(k):
        
        if i in indice_max: 
            ax1.bar(C[i], NN[i],largeur,  color = [0.15,0.15,0.80], edgecolor="white", hatch="/", 
                    lw=1., zorder = 0,alpha=0.9)
        else:
            ax1.bar(C[i], NN[i],largeur, align='center', edgecolor="white")
        
        ax1.text(C[i], NN[i], "%d"%(NN[i]),fontsize=9, style='italic', 
                 horizontalalignment='center',verticalalignment='bottom')

    ax1.set_xticks(Ext)  ## positions des extrémités des classes
    ax1.set_xticklabels(TT ,fontsize=10, rotation=45)
    ax1.set_xlim(numpy.min(data)-0.75*largeur, numpy.max(data)+0.75*largeur)
    ax1.set_ylim(0.0, numpy.max(NN)+3.0)
    ax1.set_xlabel("Valeurs",fontsize=13,labelpad=0)
    ax1.set_ylabel("Effectifs",fontsize=14)
    
    if nom is None:
        plt.show()
    else:
        nom_fig=str(nom)+".png"
        plt.savefig(nom_fig, format="png")
        plt.close()

def Histo_Continue_Groupee(Ext_G,Ext_D,Effectifs):
    import numpy
    import matplotlib.pyplot as plt
    
    nb_intervalle=len(Effectifs)
        
    centres=[0.5*(Ext_G[k]+Ext_D[k])   for k in range(nb_intervalle)]
    
    fig,ax=plt.subplots(figsize=(11,8))

    # haut et droit
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ticks
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks([])
    
    ## commande plt.bar
    ## matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
    ## pour text
    ## matplotlib.pyplot.text(x, y, s, fontdict=None, **kwargs)
    
    for k in range(nb_intervalle):
        largeur=Ext_D[k]-Ext_G[k]
        plt.bar(centres[k],Effectifs[k], width=largeur,ec="white",color=numpy.random.rand(3))
        plt.text(centres[k],Effectifs[k],Effectifs[k], horizontalalignment='center',verticalalignment='bottom')
    
    postion_xticks=Ext_G
    postion_xticks.append(Ext_D[nb_intervalle-1])
    Ticks_format = ["{:.4f}".format(c) if numpy.abs(c-numpy.round(c))!=0 else "{:d}".format(int(c)) for c in postion_xticks]
  
    #plt.xticks(centres,["{:.4f}".format(c) for c in centres], rotation=45)
    
    plt.xticks(postion_xticks,Ticks_format, rotation=45)
    
    plt.xlabel("Valeurs", fontsize=12)
    plt.ylabel("Fréquences", fontsize=15)
    plt.show()
    
def stats_descriptives(data):
    import numpy
    import math
    
    # on transforme les données en liste
    data=[x for x in data]
    
    Resultats={}
    n=len(data)
    # minimum
    m0=min(data)
    # maximum
    m1=max(data)
    # moyenne
    m=Moments_r(data,1)
    # variance
    sigma2=Moments_Centre_r(data,2)
    # Ecart-type
    sigma=math.sqrt(sigma2)
    
    gamma1=Moments_Centre_r(data,3)/sigma**3
    gamma2=-3.0+Moments_Centre_r(data,4)/sigma**4
    
    Resultats["nombre d'observations"]=n
    Resultats["minimum"]=numpy.round(m0,4)
    Resultats["maximum"]=numpy.round(m1,4)
    Resultats["moyenne"]=numpy.round(m,4)
    Resultats["variance"]=numpy.round(sigma2,4)
    Resultats["coefficient d'asymétrie (skewness)"]=numpy.round(gamma1,4)
    Resultats["coefficient d’aplatissement (kurtosis)"]=numpy.round(gamma2,4)
    
    return Resultats
    

def Regerssion_table_html(x_data,y_data,nom):
      nom_fig=nom+".png"
      nom_html=nom+".html"

      import matplotlib.pyplot as plt
      plt.close("all")
      import numpy

      import webbrowser
      import os
      chemin=os.getcwd()+"\\"+nom_html
      #print(chemin)

      def retoune_entier(x):
            import math
            if math.fabs(x-int(x))==0:
                  return int(x)
            else:
                  return float("{:.4f}".format(x))

      # on ordonne les données
      indice=numpy.argsort(x_data)
      x_data=[t for t in sorted(x_data)]
      y_data=[y_data[k] for k in indice]

      x_bar=numpy.mean(x_data)
      y_bar=numpy.mean(y_data)

      S_x=[(x-x_bar)**2 for x in x_data]
      S_xy=[(x-x_bar)*(y-y_bar) for x,y in zip(x_data,y_data)]

      # Estimateurs de a et b
      b_chap=numpy.sum(S_xy)/(1.0*numpy.sum(S_x))
      a_chap=y_bar-b_chap*x_bar
      #print(a_chap)
      #print(b_chap)

      # grpahe
      plt.plot(x_data,y_data,'o', label="observée")
      plt.plot(x_data,[a_chap+b_chap*t  for t in x_data],lw=2,color=[1,0,0], label="estimée")
      plt.title("Y={}+{} X".format(retoune_entier(a_chap),retoune_entier(b_chap)))
      plt.xlabel("X")
      plt.ylabel("Y")
      plt.legend(loc='best')
      
      plt.savefig("%s"%(nom_fig),format='png', dpi=400)
      #plt.show()
      plt.close()

      # Ecrirture en format html

      file = open(nom_html,"w") 

      file.write("<!DOCTYPE html>\n")
      file.write("<html amp lang=\"en\">\n")

      file.write("<head>\n")
      file.write("<meta charset=\"utf-8\">\n")

      file.write("<link rel=\"stylesheet\" type=\"text/css\" href=\"Mon_style.css\">\n")

      file.write("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>\n")
      file.write("<title>\n")

      file.write("R&eacute;gression lin&eacute;aire\n")
      file.write("</title>\n")

      file.write("<link rel=\"icon\" href=\"pi.ico\" />\n")

      file.write("</head>\n")

      file.write("<body>\n")
      file.write("<br></br>\n")
      file.write("<h1>  R&eacute;gression lin&eacute;aire </h1>\n")

      file.write("<br></br>\n")

      # Ecriture de la table

      file.write("<table id=\"tableA\">\n")
      file.write("<tr>\n")
      file.write(r"<th>$$x_{i}$$</th>")
      file.write("\n")
      file.write(r"<th>$$y_{i}$$</th>")
      file.write("\n")
      file.write(r"<th>$$(x_{i}-\overline{x}_{n})^2$$</th>")
      file.write("\n")
      file.write(r"<th>$$(x_{i}-\overline{x}_{n})(y_{i}-\overline{y}_{n})$$</th>")
      file.write("\n")
      file.write("</tr>\n")
      
      for k in range(len(x_data)):
            file.write("<tr>\n")
            file.write("<td>%s</td> \n"%retoune_entier(x_data[k]))
            file.write("<td>%s</td> \n"%retoune_entier(y_data[k]))
            file.write("<td>%.4f</td> \n"%(S_x[k]))
            file.write("<td>%.4f</td> \n"%(S_xy[k]))
            file.write("</tr> \n")

      file.write("<tr> \n")
      file.write(r"<td>$$\sum x_{i} =%.4f$$</td>"%(numpy.sum(x_data)))
      file.write("\n")
      file.write(r"<td>$$\sum y_{i} =%.4f$$</td>"%(numpy.sum(y_data)))
      file.write("\n")
      file.write(r"<td>$$\sum (x_{i}-\overline{x}_{n})^2 =%.4f$$</td>"%(numpy.sum(S_x)))
      file.write("\n")
      file.write(r"<td>$$\sum  (x_{i}-\overline{x}_{n})(y_{i}-\overline{y}_{n})=%.4f$$</td>"%(numpy.sum(S_xy)))
      file.write("\n")
      file.write("</tr> \n")
      

      file.write("</table>\n")

      file.write("<br></br>\n")

      # insérer la figure

      file.write("<center>\n")

      file.write("<figure>\n")
      file.write("<img src={}\n".format(nom_fig))
      
      file.write("width=\"500\" height=\"400\"\n")
      file.write("alt=\"Droite Regeression\">\n")
      file.write("<figcaption> Estimation des param&egrave;tres de la droite de r&eacute;gression</figcaption>\n")
      file.write("</figure>\n")

      file.write("</center>\n")

      # fin insérer la figure

      file.write("<br></br>\n")

      file.write("</body> \n")
      file.write("</html> \n")

      file.close()

      


      ### Affichage html
      webbrowser.get().open(nom_html)  ## ceci fonctionne sous windows
      #webbrowser.get().open(chemin)  ## voir si ceci fonctionne sous mac



###### Données test
####X=[0.99,1.02,1.15,1.29,1.46,1.36,0.87,1.23,1.55,1.40,1.19,
####   1.15,0.98,1.01,1.11,1.20,1.26,1.32,1.43,0.95]
####
####Y=[90.01,89.05,91.43,93.74,96.73,94.45,87.59,91.77,99.42,93.65,
####   93.54,92.52,90.56,89.54,89.85,90.39,93.25,93.41,94.98,87.33]
####
####reg_table.Regerssion_table_html(X,Y,"nom_du_fichier_html")

def Regression_mutiple(X_data,Y_data,Constante=True):
    import numpy
    Y_data=numpy.array([t for t in Y_data])
    N_L=len(Y_data)
    # c r é a t i o n de l a matrice M
    # avec l a c on s ta n te a
    if Constante is True:
        M=numpy.vstack((numpy.ones(N_L),X_data[0]))
        for j in range(1,len(X_data)):
            M=numpy.vstack((M,X_data[j]))
    # sans l a c on s ta n te a
    else :
        M=X_data[0]
        for j in range(1,len(X_data)):
            M=numpy.vstack((M,X_data[j]))
    M=M.T
    S1=numpy.matmul(M.T, M)
    S2=numpy.linalg.inv(S1)
    S3=numpy.matmul(Y_data.T, M)
    S=numpy.matmul(S2, S3) # e st i m a te u r
    print(S)
    return list(S)

#########################################################################
### histogramme Histogramme des données Discrètes en base64 pour flask
#########################################################################

def Histo_Discrete64(data):
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('Agg')
    plt.switch_backend('agg')
    from base64 import b64encode
    import os
    
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
    
    # sous fonction pour compter les occurrences
    def comptage(data):
        data=sorted(data)
        Dic_compt={}
        for valeur in data:
            Dic_compt[valeur]=data.count(valeur)
        return Dic_compt

    D=comptage(data)
    
    valeurs=[k for k in D.keys()]
    effectifs=[v for v in D.values()]
    i_mode=numpy.argmax(effectifs)
    ### multi_mode
    indice_mode=[i for i in range(len(effectifs)) if effectifs[i]==effectifs[i_mode]]

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)

    #ax1 =plt.gca()
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    
    ax1.set_xticks(valeurs)  ## positions des extrémités des classes
    ax1.set_xticklabels(valeurs ,fontsize=10, rotation=25)
    ax1.set_xlabel("Valeurs",fontsize=14)
    ax1.set_ylabel("Effectifes",fontsize=14)
    
    for k in range(len(valeurs)):
        if k not in indice_mode:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",color = numpy.random.rand(3))
        else:
            plt.bar(valeurs[k], height= effectifs[k],edgecolor="white",
                    color = [0.15,0.15,0.85],hatch="X", lw=1., zorder = 0)
        for i in range(len(valeurs)):
            ax1.text(valeurs[i], effectifs[i], "%d"%(effectifs[i]),fontsize=9,
                     horizontalalignment='center',verticalalignment='bottom',style='italic')
    
    # nom figure
    plt.savefig('histo64.png')
    plt.close()
    plot_file = open('histo64.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("histo64.png")

    return img_elem

#####################################################################
### Histogramme des données Continues en base64 pour flask 
#####################################################################

def Histo_Continue64(data,k):
    # k=nombre d'intervalles
    
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib
    #matplotlib.use('Agg')
    plt.switch_backend('agg')
    from base64 import b64encode
    import os
    
    plt.rcParams['hatch.color'] = [0.9,0.9,0.9]
    
    data=numpy.array([x for x in data])
    Ext=[min(data)+(max(data)-min(data))*i/(1.0*k) for i in range(k+1)]
    C=[0.5*(Ext[i]+Ext[i+1]) for i in range(k)]

    NN=[] # Effectifs des classes
    for i in range(k):
        NN.append(((Ext[i] <= data) & (data<=Ext[i+1])).sum())
        
    # pour la classe modale
    indice_max=[i for i in range(k) if NN[i]==numpy.max(NN)]
    
    TT=[str("{:.4f}".format(t)) for t in Ext]  ## pour afficher les extrémités des classes
    
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)
    # cacher le cadre
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks([])
    largeur=Ext[1]-Ext[0]  #  largeur des classes
    
    for i in range(k):
        
        if i in indice_max: 
            ax1.bar(C[i], NN[i],largeur,  color = [0.15,0.15,0.80], edgecolor="white", hatch="/", 
                    lw=1., zorder = 0,alpha=0.9)
        else:
            ax1.bar(C[i], NN[i],largeur, align='center', color = numpy.random.rand(3),edgecolor="white")
        
        ax1.text(C[i], NN[i], "%d"%(NN[i]),fontsize=9, style='italic', 
                 horizontalalignment='center',verticalalignment='bottom')

    plt.axvline(x = numpy.mean(data), color = 'red',lw=4, label = 'valeur moyenne')
    ax1.set_xticks(Ext)  ## positions des extrémités des classes
    ax1.set_xticklabels(TT ,fontsize=9, rotation=45)
    ax1.set_xlim(numpy.min(data)-0.75*largeur, numpy.max(data)+0.75*largeur)
    ax1.set_ylim(0.0, numpy.max(NN)+3.0)
    ax1.set_xlabel("Valeurs",fontsize=13,labelpad=0)
    ax1.set_ylabel("Effectifs",fontsize=14)

    plt.legend(loc='upper left')

    # nom figure
    plt.savefig('histo64.png')
    plt.close()
    plot_file = open('histo64.png', 'rb')
    base64_string = b64encode(plot_file.read()).decode()
    plot_file.close()


    img_elem = "{}".format(base64_string)

    os.remove("histo64.png")

    return img_elem
