from flask import Flask, render_template, request, url_for , redirect

# Libs
from Lib_fonctions import*
from Lib_Graphe_Asymetriques import*
from Lib_EMV_Asymetriques import*

### Données
import ssl
import pandas
ssl._create_default_https_context = ssl._create_unverified_context

site_Transilien="https://data.sncf.com/api/explore/v2.1/catalog/datasets/ponctualite-mensuelle-transilien/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"

df_Transilien = pandas.read_csv(site_Transilien,sep=';')

# Ordonner les données par 'Ligne' et 'Date'

data_Transilien_1=df_Transilien.sort_values(by=[df_Transilien.columns[2],df_Transilien.columns[0]])
indice_Sans_A=data_Transilien_1[data_Transilien_1[data_Transilien_1.columns[3]]=="RER A"].index
data_Transilien=data_Transilien_1.drop(indice_Sans_A)

# Nom de la ligne
dict_ligne=dict(data_Transilien[data_Transilien.columns[3]].value_counts())
Liste_ligne=[L for L in dict_ligne.keys()]
Liste_ligne.append('Total')

###

app = Flask(__name__)

# page d'accueil
@app.route('/')
def index():
    return render_template('index_projet.html',Liste_ligne=Liste_ligne)

                           
@app.route('/BetaGauche', methods=['GET', 'POST'])          
def BetaGauche():
    if request.method == 'POST':
        a=float(request.form['parama'])
        b=float(request.form['paramb'])

        AAlpha=float(request.form['paramalpha'])
        BBeta=float(request.form['parambeta'])

        LLambda=float(request.form['paramlambda'])


        result_BetaG=image64_Beta(a,b,LLambda,AAlpha,BBeta,asymetrie="gauche")
        result_BetaD=image64_Beta(a,b,LLambda,AAlpha,BBeta,asymetrie="droite")


        return render_template('Resultats_graphes.html',
                               result_BetaG=result_BetaG,
                               Titre_Beta="Loi Beta tronquée",
                               result_BetaD=result_BetaD)
    else:
        return render_template('index_projet.html')
        #return redirect(url_for('index'))

@app.route('/GammaGauche', methods=['GET', 'POST'])          
def GammaGauche():
    if request.method == 'POST':
        a=float(request.form['parama'])
        b=float(request.form['paramb'])

        AAlpha=float(request.form['paramalpha'])
        BBeta=float(request.form['parambeta'])

        LLambda=float(request.form['paramlambda'])


        result_GammaG=image64_Gamma(a,b,LLambda,AAlpha,BBeta,asymetrie="gauche")
        result_GammaD=image64_Gamma(a,b,LLambda,AAlpha,BBeta,asymetrie="droite")

        return render_template('Resultats_graphes.html',
                               result_GammaG=result_GammaG,
							   Titre_Gamma="Loi Gamma tronquée",
                               result_GammaD=result_GammaD)
    else:
        return render_template('index_projet.html')


@app.route('/WeibullGauche', methods=['GET', 'POST'])          
def WeibullGauche():
    if request.method == 'POST':
        a=float(request.form['parama'])
        b=float(request.form['paramb'])

        AAlpha=float(request.form['paramalpha'])
        BBeta=float(request.form['parambeta'])

        LLambda=float(request.form['paramlambda'])


        result_WeibullG=image64_Weibull(a,b,LLambda,AAlpha,BBeta,asymetrie="gauche")
        result_WeibullD=image64_Weibull(a,b,LLambda,AAlpha,BBeta,asymetrie="droite")

        return render_template('Resultats_graphes.html',
                               result_WeibullG=result_WeibullG,
							   Titre_Weibull="Loi Weibull tronquée",
                               result_WeibullD=result_WeibullD)
    else:
        return render_template('index_projet.html')



@app.route('/Transilien', methods=['GET', 'POST'])
def Transilien():

    if request.method == 'POST':

        #choix = Nom de la ligne

        choix = request.form.get('ligne_select')
        if choix=="Total":
            data_R=data_Transilien[data_Transilien.columns[4]].dropna().values.tolist()
        else:
            data_R=data_Transilien[data_Transilien[data_Transilien.columns[3]]==choix][data_Transilien.columns[4]].dropna().values.tolist()

        # stas de data
        Stats_html=stats_descriptives(data_R)
        # histogramme
        histo_html=Histo_Continue64(data_R,20)
        # Estimation
        # ajustement par la loi Beta (asymétrie à droite)
        img_Beta_D,param_Beta_D,Irg_Beta_D=Beta_Droite_EMV64(data_R,nb_classe=20)

        # img_Beta_D : image en format Base64
        # param_Beta_D : paramètres estimés
        #Irg_Beta_D : Dictionnaire pour l'estimation de différents seuils d'irrégularité.
        

        # ajustement par la loi Gamma

        img_Gamma_D,param_Gamma_D,Irg_Gamma_D=Gamma_Droite_EMV64(data_R,nb_classe=20)

        # ajustement par la loi Weibull

        img_weibull_D,param_weibull_D,Irg_Weibull_D=Weibull_Droite_EMV64(data_R,nb_classe=20)

        
        return render_template('Resultats_Application.html',choix=choix,
                               Stats_html=Stats_html,
                               histo_html=histo_html,
                               img_Beta_D=img_Beta_D,
                               param_Beta_D=param_Beta_D,
                               Irg_Beta_D=Irg_Beta_D,
                               img_Gamma_D=img_Gamma_D,
                               param_Gamma_D=param_Gamma_D,
                               Irg_Gamma_D=Irg_Gamma_D,
                               img_weibull_D=img_weibull_D,
                               param_weibull_D=param_weibull_D,
                               Irg_Weibull_D=Irg_Weibull_D
                               )
    else:

        return redirect(url_for('index'))


""" if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:8080/")
    app.run(debug=True ,port=8080,use_reloader=False,threaded=True) """

""" if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:8080/")

    from werkzeug.serving import run_simple
    run_simple('localhost', 8080, app) """

if __name__ == '__main__':
    app.run(threaded=True)