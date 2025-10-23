# -- coding: utf-8 --
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import re
import pandas as pd
import json
import requests
from io import BytesIO

# --- Configuration de la page Streamlit et Design ---
st.set_page_config(
    page_title="✨ Visualiseur Dynamique d'Espaces d'Acceptabilité",
    page_icon="🧠",
    layout="wide"
)

# --- Fonctions de Calcul (Cœur du projet) ---

@st.cache_data
def calculate_h_categorizer(A, R, w, epsilon=1e-5):
    """Implémente la sémantique h-categorizer pondérée avec contrôle de convergence."""
    n = len(A)
    arg_to_idx = {arg: i for i, arg in enumerate(A)}
    attackers = [[] for _ in range(n)]
    for attacker, attacked in R:
        if attacker in arg_to_idx and attacked in arg_to_idx:
            attackers[arg_to_idx[attacked]].append(arg_to_idx[attacker])
    
    # 1. Initialisation
    hc_k = np.zeros(n)
    
    # 2. Itération
    for _ in range(500): # Limite de 500 itérations
        # Calcule la somme des attaquants pour chaque argument
        sum_attackers_hc = np.array([np.sum(hc_k[att]) for att in attackers])
        
        # Formule de mise à jour: hc_{k+1}(a) = w(a) / (1 + sum_{b -> a} hc_k(b))
        hc_k_plus_1 = w / (1 + sum_attackers_hc)
        
        # Vérification de la convergence
        if np.max(np.abs(hc_k_plus_1 - hc_k)) < epsilon:
            return hc_k_plus_1
        
        hc_k = hc_k_plus_1
        
    return hc_k_plus_1 # Retourne le résultat après le nombre max d'itérations


@st.cache_data
def compute_acceptability_space(A, R, num_samples):
    """Approxime l'espace d'acceptabilité (Monte Carlo) et construit l'enveloppe convexe."""
    n = len(A)
    if n == 0:
        return np.array([]), None
        
    acceptability_points = []
    # Génération de tous les poids en une seule fois
    random_weights = np.random.rand(num_samples, n)
    
    progress_bar = st.progress(0, text=f"Calcul de {num_samples} points d'acceptabilité...")
    for i, w in enumerate(random_weights):
        x_i = calculate_h_categorizer(A, R, w)
        acceptability_points.append(x_i)
        if (i + 1) % 500 == 0:
            progress_bar.progress((i + 1) / num_samples, text=f"Calcul en cours... {i+1}/{num_samples} points")

    progress_bar.empty()
    points = np.array(acceptability_points)
    
    hull = None
    # L'enveloppe convexe nécessite au moins n+1 points pour un espace de dimension n
    if n > 1 and len(points) > n:
        try:
            hull = ConvexHull(points)
        except Exception as e:
            st.warning(f"⚠️ Impossible de construire l'enveloppe convexe. Erreur: {e}")
            
    return points, hull

# --- Fonction de Visualisation (Utilise Plotly pour l'interactivité) ---

def visualize_space(A, points, hull, n):
    """Gère l'affichage des graphiques selon la dimension (n)."""
    
    # Couleurs améliorées
    colors = {
        'primary': '#6366F1',  # Violet moderne
        'secondary': '#10B981', # Vert émeraude
        'accent': '#F59E0B',   # Orange ambre
        'segment': '#8B5CF6',  # Violet vif
        'scatter': '#3B82F6',  # Bleu
        'hull': '#EF4444'      


    }
    
    # Cas n=1 : Segment (Utilise Matplotlib pour la simplicité du segment 1D)
    if n == 1:
        st.subheader(f"Segment d'acceptabilité pour l'argument '{A[0]}'")
        min_val, max_val = np.min(points), np.max(points)
        
        # Utilisation de Plotly pour le segment 1D aussi pour uniformité
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[0, 0],
            mode='lines',
            line=dict(color=colors['segment'], width=8),
            name='Segment'
        ))
        fig.update_layout(
            title=f"Espace d'Acceptabilité 1D (Intervalle: [{min_val:.3f}, {max_val:.3f}])",
            xaxis_title=f"Degré d'acceptabilité de '{A[0]}'",
            yaxis_visible=False,
            height=200,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cas n=2 : Plot 2D (Utilise Plotly pour le zoom)
    elif n == 2:
        st.subheader("Enveloppe Convexe 2D")
        
        df = pd.DataFrame(points, columns=[A[0], A[1]])
        
        fig = px.scatter(
            df, 
            x=A[0], 
            y=A[1], 
            opacity=0.4,
            color_discrete_sequence=[colors['scatter']],
            title="Points échantillonnés et Enveloppe Convexe 2D",
            labels={'x': f"Acceptabilité de '{A[0]}'", 'y': f"Acceptabilité de '{A[1]}'"}
        )
        
        if hull:
            # Ajouter le contour de l'enveloppe convexe (Simplices sont des arêtes en 2D)
            for simplex in hull.simplices:
                fig.add_trace(go.Scatter(
                    x=points[simplex, 0], 
                    y=points[simplex, 1], 
                    mode='lines', 
                    line=dict(color=colors['primary'], width=3), 
                    name='Contour'
                ))
        
        fig.update_layout(
            xaxis_range=[0, 1], 
            yaxis_range=[0, 1], 
            showlegend=False,
            height=600,
            plot_bgcolor='rgba(240,240,240,0.1)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cas n=3 : Plot 3D interactif (Utilise Plotly pour la rotation)
    elif n == 3:
        st.subheader("Enveloppe Convexe 3D (Interactive)")
        
        if hull:
            fig_3d = go.Figure(data=[
                go.Mesh3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                    opacity=0.7,
                    color=colors['primary'],
                    name='Enveloppe Convexe'
                )
            ])
        else:
            fig_3d = px.scatter_3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                opacity=0.6,
                color_discrete_sequence=[colors['secondary']],
                title="Points échantillonnés 3D"
            )
            
        fig_3d.update_layout(
            scene=dict(
                xaxis_title=A[0], yaxis_title=A[1], zaxis_title=A[2],
                xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1]),
                aspectmode='cube'
            ),
            title=f"Espace d'Acceptabilité 3D - Axes: {A[0]}, {A[1]}, {A[2]}",
            height=700
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Cas n > 3 : Tranche 3D interactive (Projection)
    elif n > 3:
        st.subheader("Tranche 3D Interactive (Projection de l'Espace à N Dimensions)")
        st.info("💡 Pour |A| > 3, on affiche la projection 3D en fixant les autres dimensions.")
        
        # Sélecteurs pour les axes à afficher
        cols = st.columns(3)
        with cols[0]:
            x_axis_idx = st.selectbox("Axe X", options=range(n), format_func=lambda i: A[i], index=0)
        with cols[1]:
            y_axis_idx = st.selectbox("Axe Y", options=range(n), format_func=lambda i: A[i], index=1)
        with cols[2]:
            z_axis_idx = st.selectbox("Axe Z", options=range(n), format_func=lambda i: A[i], index=2)
        
        axes_to_plot = [x_axis_idx, y_axis_idx, z_axis_idx]
        
        # Sliders pour les autres dimensions (Dimensions fixes)
        fixed_axes_indices = [i for i in range(n) if i not in axes_to_plot]
        fixed_values = {}
        
        if fixed_axes_indices:
            st.markdown("##### ⚙️ Contrôles des Dimensions Fixes")
            
            # Utilisation du caching pour ne pas perdre la valeur du slider entre les relancements
            if 'tolerance' not in st.session_state:
                st.session_state.tolerance = 0.05
            
            tolerance = st.slider("Tolérance de la 'tranche' (± valeur fixée)", 0.01, 0.5, st.session_state.tolerance, key='tolerance_slider')
            
            # Création de la mask de sélection
            mask = np.full(points.shape[0], True)
            
            for i in fixed_axes_indices:
                if f'fixed_value_{i}' not in st.session_state:
                    st.session_state[f'fixed_value_{i}'] = 0.5
                    
                fixed_val = st.slider(
                    f"Valeur Fixe pour '{A[i]}' **({i+1})**", 
                    0.0, 1.0, 
                    st.session_state[f'fixed_value_{i}'], 
                    0.01,
                    key=f'fixed_value_slider_{i}'
                )
                fixed_values[i] = fixed_val
                
                # Appliquer le masquage pour la tranche
                mask &= (np.abs(points[:, i] - fixed_val) < tolerance)
                
            sliced_points = points[mask]
        else:
            sliced_points = points

        # Affichage du Plotly 3D pour la tranche
        if sliced_points.shape[0] == 0:
            st.warning("Aucun point trouvé dans la tranche. Essayez d'augmenter la tolérance.")
        else:
            fig_slice = px.scatter_3d(
                x=sliced_points[:, axes_to_plot[0]],
                y=sliced_points[:, axes_to_plot[1]],
                z=sliced_points[:, axes_to_plot[2]],
                title=f"Tranche 3D de l'Espace ({len(sliced_points)} points affichés)",
                labels={
                    'x': A[axes_to_plot[0]], 
                    'y': A[axes_to_plot[1]], 
                    'z': A[axes_to_plot[2]]
                },
                opacity=0.6,
                color_discrete_sequence=[colors['accent']]
            )
            fig_slice.update_layout(
                scene=dict(
                    xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), zaxis=dict(range=[0, 1]),
                    aspectmode='cube'
                ),
                height=700
            )
            st.plotly_chart(fig_slice, use_container_width=True)

# --- Interface Utilisateur Principale (avec Onglets) ---

st.title("🧠 Visualiseur Dynamique des Espaces d'Acceptabilité")
st.write("""
Cette application implémente et visualise la sémantique **h-categorizer pondérée** ($\\Sigma_{\\mathrm{HC}}$) pour l'argumentation graduelle, 
en utilisant une méthode d'échantillonnage Monte Carlo pour approximer l'espace d'acceptabilité.
""")

# --- Barre Latérale (Design et Documentation) ---
st.sidebar.header("Configuration du Graphe")

# Lien vers le Notebook de Documentation
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Documentation Détaillée")

# Définition du graphe par l'utilisateur
default_args = "a, b, c, d"
default_attacks = "a -> b\nb -> c\nc -> a\na -> d"

args_input = st.sidebar.text_area("1. Liste des arguments (séparés par des virgules)", value=default_args, key='args_input')
attacks_input = st.sidebar.text_area("2. Liste des attaques (une par ligne, format: attaquant -> attaque)", value=default_attacks, key='attacks_input')

# Parsing des entrées utilisateur
A = sorted([arg.strip() for arg in args_input.split(',') if arg.strip()])
R = []
for line in attacks_input.split('\n'):
    match = re.match(r'(\w+)\s*->\s*(\w+)', line.strip())
    if match:
        attacker, attacked = match.groups()
        if attacker in A and attacked in A:
            R.append((attacker, attacked))

st.sidebar.markdown(f"*Arguments détectés* : **{len(A)}**")
st.sidebar.markdown(f"*Attaques détectées* : **{len(R)}**")

num_samples = st.sidebar.slider(
    "3. Nombre de points à échantillonner",
    min_value=1000,
    max_value=150000, # Augmenté pour l'excellence
    value=50000,
    step=1000,
    help="Plus le nombre est élevé, plus l'approximation de l'espace est précise, mais le calcul est long.",
    key='num_samples_slider'
)

# --- Contenu des Onglets ---
tab_main, tab_notebook = st.tabs(["🚀 Configuration & Résultats", "📚 Notebook"])

with tab_main:
    st.header("Configuration et Visualisation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Configuration")
        st.markdown(f"""
        **Paramètres actuels :**
        - Nombre d'Arguments $|A|$ : **{len(A)}**
        - Nombre de Points Échantillonnés : **{num_samples}**
        """)
        
        if st.button("✨ Lancer le Calcul", type="primary", use_container_width=True):
            st.session_state.run_visualization = True
    
    # Affichage automatique des résultats si le calcul a été lancé
    if 'run_visualization' in st.session_state and st.session_state.run_visualization:
        n = len(A)
        if n == 0:
            st.error("Veuillez définir au moins un argument dans la barre latérale.")
        else:
            with st.spinner(f"Calcul de l'espace pour {n} arguments avec {num_samples} points..."):
                points, hull = compute_acceptability_space(A, R, num_samples)
            
            with col2:
                st.subheader("📊 Résultats")
                st.success("✅ Calcul terminé avec succès !")
                visualize_space(A, points, hull, n)
                
                if hull:
                    st.info(f"📊 Résultats : {len(points)} points générés. Enveloppe Convexe construite avec {len(hull.simplices)} simplexes.")
                elif len(points) > 0:
                    st.info(f"📊 Résultats : {len(points)} points générés. L'enveloppe convexe n'a pas été construite (voir avertissement).")
    else:
        with col2:
            st.info("👆 Configurez votre graphe dans la barre latérale et cliquez sur 'Lancer le Calcul' pour voir les résultats ici.")

with tab_notebook:
    st.markdown("""
    ## 📚 Cahier de Réponse Détaillé : Espaces d'Acceptabilité pour la Sémantique Graduelle

    Ce document présente l'implémentation et l'analyse de la sémantique h-categorizer pondérée ($\\Sigma_{HC}$), 
    ainsi qu'une méthode d'approximation et de visualisation de l'espace des degrés d'acceptabilité pour un système d'argumentation donné.
    """)
    
    # --- Question 1 ---
    st.markdown("""
    ## 1. Implémentation de la Sémantique H-Categorizer Pondérée ($\\Sigma_{HC}$) (Question 1)

    La sémantique h-categorizer ($\\Sigma_{HC}$) pour un Framework d'Argumentation Pondéré ($\\mathcal{F}=(A,R,w)$) est définie par une séquence de fonctions $hc_k : A \\to [0,1]$ donnée par la formule récursive :

    $$
    \\forall a \\in A, \\quad hc_{k+1}(a) = \\frac{w(a)}{1 + \\sum_{b \\to a} hc_k(b)}
    $$

    Où $w(a) \\in [0,1]$ est le poids de l'argument $a$, et la somme est effectuée sur tous les attaquants $b$ de $a$.
    """)
    
    with st.expander("🔬 **Code d'Implémentation et Analyse de la Convergence**"):
        st.markdown("""
        **Implémentation de la fonction calculate_h_categorizer :**
        ```python
        def calculate_h_categorizer(A, R, w, epsilon=1e-5, max_iter=500):
            n = len(A)
            arg_to_idx = {arg: i for i, arg in enumerate(A)}
            attackers = [[] for _ in range(n)]
            
            # Prétraitement des attaques pour un accès rapide
            for attacker, attacked in R:
                if attacker in arg_to_idx and attacked in arg_to_idx:
                    attackers[arg_to_idx[attacked]].append(arg_to_idx[attacker])
            
            hc_k = np.zeros(n)
            
            for k in range(max_iter):
                # Calcul des sommes d'attaquants pour chaque argument
                sum_attackers_hc = np.array([np.sum(hc_k[att]) for att in attackers])
                
                # Application de la formule du h-categorizer
                hc_k_plus_1 = w / (1 + sum_attackers_hc)
                
                # Vérification de la convergence
                if np.max(np.abs(hc_k_plus_1 - hc_k)) < epsilon:
                    return hc_k_plus_1
                
                hc_k = hc_k_plus_1
                
            return hc_k_plus_1
        ```
        
        **Analyse de la convergence :** La sémantique $\\Sigma_{HC}$ est garantie de converger en raison de la nature non expansive de sa fonction de mise à jour (elle est une contraction sur l'espace $[0,1]^n$). La boucle implémentée s'arrête lorsque la différence maximale entre les valeurs d'acceptabilité de l'itération $k$ et $k+1$ devient inférieure au seuil $\\epsilon$, ou si le nombre maximal d'itérations est atteint.
        """)
    
    # --- Question 2 ---
    st.markdown("""
    ## 2. Approximation de l'Espace d'Acceptabilité (Question 2)

    L'espace des degrés d'acceptabilité est l'ensemble $S_{\\Sigma_{HC}}(\\mathcal{F}) = \\{\\Sigma_{HC}(A,R,w) \\mid w \\in [0,1]^{|A|}\\}$. Cet espace est un sous-ensemble convexe (bien que non nécessairement polyédrique) de $[0,1]^{|A|}$.

    Nous utilisons une méthode Monte Carlo pour approximer cet espace, en échantillonnant un grand nombre de vecteurs de poids aléatoires ($w_i$) et en calculant les vecteurs d'acceptabilité correspondants ($x_i$).
    """)
    
    with st.expander("🔧 **Code d'Approximation et Construction de l'Enveloppe Convexe**"):
        st.markdown("""
        **Fonction compute_acceptability_space :**
        ```python
        def compute_acceptability_space(A, R, num_samples):
            n = len(A)
            if n == 0:
                return np.array([]), None
                
            acceptability_points = []
            
            # Échantillonnage des vecteurs w_i
            random_weights = np.random.rand(num_samples, n)
            
            # Calcul des x_i correspondants
            for w in random_weights:
                x_i = calculate_h_categorizer(A, R, w)
                acceptability_points.append(x_i)

            points = np.array(acceptability_points)
            
            # Construction de l'Enveloppe Convexe
            hull = None
            if n > 1 and len(points) > n:
                try:
                    hull = ConvexHull(points)
                except Exception as e:
                    print(f"ATTENTION: Impossible de construire l'enveloppe convexe. Erreur: {e}")
                    
            return points, hull
        ```
        
        **Justification :** L'approximation par l'Enveloppe Convexe est pertinente car, en raison de la nature continue et de la convexité de l'espace $S_{\\Sigma_{HC}}(\\mathcal{F})$, les points générés sont distribués à l'intérieur (et sur la frontière) de cet espace. L'enveloppe convexe des points échantillonnés offre donc une excellente approximation polyédrique de l'espace réel.
        """)
    
    # --- Question 3 ---
    st.markdown("""
    ## 3. Visualisation de l'Enveloppe Convexe (Question 3)

    La visualisation s'adapte à la dimension de l'espace ($|A|=n$). Les visualisations utilisent Plotly pour garantir l'interactivité.
    """)
    
    with st.expander("🎨 **Stratégies de Visualisation par Dimension**"):
        st.markdown("""
        - **$|A| = 1$** : Affichage d'un segment représentant l'intervalle des degrés d'acceptabilité possibles
        - **$|A| = 2$** : Graphique 2D avec nuage de points et enveloppe convexe
        - **$|A| = 3$** : Visualisation 3D interactive avec mesh de l'enveloppe convexe
        - **$|A| > 3$** : Tranche 3D avec sélection des axes et contrôle des dimensions fixes via sliders
        """)
    
   # --- Exemples avec Images ---
    st.markdown("""
    ## 🧪 Exemples et Résultats

    Voici quelques exemples de graphes testés avec leurs visualisations correspondantes :
    """)
    
    with st.expander("📊 **Exemples de Graphes Testés avec Résultats**"):
        st.markdown("""
        ### **Cycle à 3 arguments :**
        - $A = \\{a, b, c\\}$
        - $R = \\{a \\to b, b \\to c, c \\to a\\}$
        - **Résultat :** Un espace 3D avec une structure symétrique caractéristique des cycles
        """)
        
        # Méthode 1 : Charger l'image locale
        try:
            st.image("A=3.png", caption="Visualisation 3D - Cycle à 3 arguments", use_column_width=True)
        except:
            st.warning("Image A=3.png non trouvée. Assurez-vous qu'elle est dans le même dossier que l'application.")
        
        st.markdown("---")
        
        st.markdown("""
        ### **Graphe simple à 2 arguments :**
        - $A = \\{a, b\\}$
        - $R = \\{a \\to b\\}$
        - **Résultat :** Un espace 2D montrant la relation entre les degrés d'acceptabilité des deux arguments
        """)
        
        try:
            st.image("A=2.png", caption="Visualisation 2D - Graphe à 2 arguments", use_column_width=True)
        except:
            st.warning("Image A=2.png non trouvée")
        
        st.markdown("---")
        
        st.markdown("""
        ### **Graphe à 4 arguments :**
        - $A = \\{a, b, c, d\\}$
        - $R = \\{a \\to b, b \\to c, c \\to d, d \\to a\\}$
        - **Résultat :** Visualisation par tranches 3D de l'espace à 4 dimensions
        """)
        
        try:
            st.image("A=4.png", caption="Visualisation 3D - Graphe à 4 arguments", use_column_width=True)
        except:
            st.warning("Image A=4.png non trouvée")
    
    # --- Téléchargement du vrai notebook Colab ---
    st.markdown("---")
    
    st.markdown("### 📥 Télécharger le Notebook Complet")
    
    # Lien direct vers le notebook Colab
    colab_url = "https://colab.research.google.com/drive/1_vA9JZc7E6zs__OXmji7Sq4Bz-8KI_IF?usp=sharing"
    
    st.markdown(f"""
    <div style='border: 2px solid #6366F1; border-radius: 10px; padding: 20px; background-color: #f8f9fa;'>
        <h4 style='color: #6366F1;'>📖 Notebook Colab Complet</h4>
        <p>Accédez au notebook complet contenant tous les codes, analyses et visualisations :</p>
        <a href='{colab_url}' target='_blank' style='display: inline-block; padding: 10px 20px; background-color: #6366F1; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0;'>
            🔗 Ouvrir dans Google Colab
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton de téléchargement alternatif
    st.download_button(
        label="📥 Télécharger le Notebook (.ipynb)",
        data="",  # Vous pouvez ajouter le contenu réel du notebook ici si disponible
        file_name="cahier_reponse_espaces_acceptabilite.ipynb",
        mime="application/x-ipynb+json",
        type="primary",
        use_container_width=True,
        disabled=True,  # Désactivé car nous utilisons le lien Colab
        help="Le notebook complet est disponible via le lien Google Colab ci-dessus"
    )

# Initialisation de l'état de session
if 'run_visualization' not in st.session_state:
    st.session_state.run_visualization = False