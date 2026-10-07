# Py_OLPOS : Python Online Portfolio Optimisation Strategies

**Projet académique de Machine Learning et Data Science**  
*Auteurs : Marouan Maghzaoui & Tudor Haruta*

## 📌 Contexte et Objectifs
La gestion d'un portefeuille financier est un défi complexe pour les algorithmes statistiques et le Machine Learning. Ce projet a pour but de concevoir un modèle d'allocation de portefeuille "online" automatisé, capable d'effectuer des prédictions d'achat et de vente d'actifs au jour le jour pour maximiser les gains, en s'affranchissant des biais émotionnels humains.

**Objectif principal :** Créer un modèle de classification capable de prédire et trier le rang de rentabilité de différentes actions pour le jour suivant, afin d'adapter dynamiquement la répartition du capital dans le portefeuille.

## 📊 Données et Ingénierie des Features
- **Sources :** Historiques de prix via l'API Yahoo Finance (Ouverture, Fermeture, Max, Min).
- **Actifs suivis :** 
  - *Actions :* Apple, Amazon, Pfizer, Google, Tesla, Facebook.
  - *Indices & Macro :* NASDAQ, NYSE, VIX (volatilité), DXY (force du dollar), paires de devises (EUR/USD, BTC/USD), Or et Pétrole.
- **Volume de données :** Environ 6 157 jours d'historique exploités (à partir de l'introduction en bourse d'Amazon).
- **Variable Cible (Target) :** Face aux limites d'une approche par régression, le problème a été redéfini en classification. La cible est une combinaison linéaire modélisant le classement journalier des gains de 3 actions piliers (Apple, Pfizer, Amazon), répartie sur 6 classes distinctes.

## 🤖 Modélisation et Optimisation
Les modèles ont été évalués selon deux critères : le score de classification (**Accuracy**) et une fonction personnalisée **`value_wallet`** simulant la valeur financière finale du portefeuille (répartition pondérée du capital : 4/7, 2/7, 1/7 selon les prédictions).

Plusieurs algorithmes ont été testés (Logistic Regression, PAC, Perceptron, SVM, Gradient Boosting). Les deux modèles retenus sur la ligne de Pareto sont :
1. **K-Nearest Neighbors (KNN : n=31, p=2) :** 
   - Meilleur score d'accuracy (0.22, supérieur à l'aléatoire fixé à 0.166). 
   - *Analyse SHAP :* Le modèle s'appuie massivement sur le volume d'échange du Bitcoin.
2. **Random Forest (n_estimators=100, min_samples_leaf=39) :** 
   - Meilleure performance financière (Portefeuille final de 20 896 $pour un investissement initial de 10 000$).
   - *Analyse SHAP :* Influence plus diversifiée des features (Google, VIX, Pfizer, NASDAQ).

## 🚀 Bilan et Pistes d'Amélioration
Bien que les scores d'accuracy restent modestes, la stratégie automatisée a permis de doubler la valeur du portefeuille initial dans l'environnement de test (bien que loin de la valeur optimale théorique de 176 286 $).

**Limites et axes de développement futurs :**
- **Frais de transaction :** Le simulateur n'intègre pas encore les taxes et frais de courtage. Lors d'un rééquilibrage journalier, ces frais impacteraient drastiquement la rentabilité réelle.
- **Deep Learning :** Explorer des architectures de réseaux de neurones pour améliorer la capacité prédictive.
- **Stratégie d'allocation :** Affiner l'algorithme de répartition mathématique du capital en fonction des rangs prédits.

---
⚠️ *Avertissement : Ce projet est purement éducatif. Les algorithmes et résultats présentés ici ne constituent en aucun cas des conseils en investissement financier ou en trading.*
