# Modèle Black-Litterman

## Introduction
Développé par Fischer Black et Robert Litterman chez Goldman Sachs en 1990,
ce modèle résout les problèmes pratiques du modèle de Markowitz.

## Problèmes Résolus

### Issues du Modèle de Markowitz:
1. **Allocations Extrêmes**: Poids trop concentrés
2. **Instabilité**: Petits changements → grandes variations
3. **Pas de Point de Départ**: Commencer avec une page blanche

### Solutions de Black-Litterman:
1. **Équilibre du Marché**: Point de départ cohérent
2. **Intégration de Vues**: Incorpore vos convictions
3. **Allocations Stables**: Résultats plus réalistes

## Principe de Fonctionnement

### 1. Équilibre du Marché (Prior)
Commence avec les capitalisations boursières comme poids initiaux.

**Formule:**
```
Π = λ * Σ * w_market
```
Où:
- Π = rendements implicites d'équilibre
- λ = coefficient d'aversion au risque
- Σ = matrice de covariance
- w_market = poids de marché

### 2. Vues de l'Investisseur

**Types de vues:**
- **Absolues**: "L'action A aura 12% de rendement"
- **Relatives**: "A surperformera B de 3%"

**Formule des vues:**
```
P * μ = Q + ε
```
Où:
- P = matrice pick (définit les actifs)
- μ = rendements attendus
- Q = vues quantifiées
- ε = incertitude des vues

### 3. Combinaison (Posterior)

Combine l'équilibre et les vues via théorème de Bayes:

```
E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ * [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

Où:
- τ = paramètre de scaling (ex: 0.05)
- Ω = matrice d'incertitude des vues

## Exemple Pratique

### Données:
- Actions: AAPL, MSFT, GOOGL
- Poids de marché: 40%, 35%, 25%

### Vues:
1. "AAPL va surperformer de 5%" (confiance: 80%)
2. "MSFT va mieux faire que GOOGL de 3%" (confiance: 60%)

### Résultat:
Black-Litterman ajuste les poids pour refléter ces vues
tout en restant proche de l'équilibre de marché.

## Avantages

1. **Point de Départ Solide**: Équilibre de marché
2. **Flexibilité**: Ajoutez vos insights
3. **Stabilité**: Allocations plus réalistes
4. **Interprétabilité**: Compréhension intuitive

## Paramètres Clés

### τ (Tau)
- Mesure l'incertitude sur l'équilibre
- Typiquement: 0.01 à 0.05
- Plus grand τ = plus d'importance aux vues

### Confiance des Vues
- Détermine le poids des vues
- Haute confiance = plus d'impact
- Faible confiance = reste proche équilibre

### Taux Sans Risque
- Base pour calculer les rendements excédentaires
- Utiliser taux des bons du Trésor

## Dans PyManager

1. Portfolio Builder → "Black-Litterman"
2. Entrez vos vues (optionnel)
3. Ajustez les paramètres (τ, confiance)
4. Le modèle calcule les poids optimaux

**Sans vues:** Retourne l'équilibre de marché pur.
**Avec vues:** Ajuste selon vos convictions.
