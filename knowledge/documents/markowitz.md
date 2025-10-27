# Théorie Moderne du Portfolio (Markowitz)

## Histoire
Développée par Harry Markowitz en 1952, cette théorie révolutionnaire
lui a valu le Prix Nobel d'Économie en 1990.

## Concept Clé
Optimiser le ratio risque/rendement en construisant un portfolio efficient.

## Hypothèses Fondamentales

1. **Investisseurs Rationnels**
   - Averses au risque
   - Préfèrent plus de rendement à moins
   - Maximisent l'utilité espérée

2. **Mesure du Risque**
   - Variance (écart-type) des rendements
   - Corrélation entre actifs
   - Matrice de covariance

3. **Efficience des Marchés**
   - Tous les investisseurs ont les mêmes informations
   - Pas de coûts de transaction
   - Actifs divisibles

## Frontière Efficiente

Ensemble des portfolios offrant:
- Le rendement maximal pour un niveau de risque donné
- Le risque minimal pour un rendement donné

## Formule Mathématique

**Rendement du Portfolio:**
```
E(Rp) = Σ wi * E(Ri)
```

**Risque du Portfolio:**
```
σp² = Σ Σ wi * wj * σij
```

Où:
- wi, wj = poids des actifs i et j
- E(Ri) = rendement espéré de l'actif i
- σij = covariance entre actifs i et j

## Portfolio Optimal

Trois stratégies principales:

### 1. Maximum Sharpe Ratio
- Meilleur rendement ajusté au risque
- Point tangent sur la frontière efficiente
- Recommandé pour la plupart des investisseurs

### 2. Minimum Variance
- Risque minimal
- Convient aux investisseurs très conservateurs
- Peut avoir un rendement faible

### 3. Maximum Return
- Rendement maximal
- Risque élevé
- Pour investisseurs agressifs

## Limites du Modèle

1. **Estimation des Paramètres**
   - Difficile d'estimer rendements futurs
   - Matrices de covariance instables
   - Sensible aux données historiques

2. **Hypothèses Irréalistes**
   - Marchés pas toujours efficients
   - Coûts de transaction existent
   - Distribution non normale

3. **Solutions Extrêmes**
   - Poids parfois irréalistes
   - Positions concentrées
   - Instabilité des allocations

## Améliorations Modernes

- **Black-Litterman**: Intègre les vues d'experts
- **Robust Optimization**: Gère l'incertitude
- **Risk Parity**: Équilibre les contributions au risque

## Dans PyManager

Utilisez le Portfolio Builder:
1. Sélectionnez vos actifs
2. Choisissez "Markowitz"
3. Sélectionnez la stratégie (Sharpe/Risk/Return)
4. Le système calcule les poids optimaux
