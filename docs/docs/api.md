# API

Cette API permet de prédire l’espèce d’une fleur d’Iris à partir de ses caractéristiques mesurées en centimètres.
Elle expose actuellement un seul endpoint : `POST /predict`.

## POST /predict — Prédire l’espèce d’Iris

Prend en entrée les quatre mesures principales d’un iris et renvoie la prédiction du modèle de machine learning entraîné.

### Description

Ce point d’entrée effectue :

- une validation des données,
- une mise en forme pour le modèle,
- une prédiction parmi les classes : `0`, `1` ou `2`

### Corps de requête (application/json)

#### Exemple

```json
{
  "sepal_length_cm": 0,
  "sepal_width_cm": 0,
  "petal_length_cm": 0,
  "petal_width_cm": 0
}
```

#### Schéma attendu

| Champ           | Type  | Description             |
| --------------- | ----- | ----------------------- |
| sepal_length_cm | float | Longueur du sépale (cm) |
| sepal_width_cm  | float | Largeur du sépale (cm)  |
| petal_length_cm | float | Longueur du pétale (cm) |
| petal_width_cm  | float | Largeur du pétale (cm)  |

Tous les champs sont **obligatoires**.

### Réponse

#### Exemple de réponse (200 — OK)

```json
{
  "prediction": 0
}
```

#### Description du contenu

| Champ      | Type | Description    |
| ---------- | ---- | -------------- |
| prediction | int  | Espèce prédite |

### Erreurs possibles

#### 400 — Données invalides

```json
{
  "error": "Invalid payload. Missing or incorrect fields."
}
```

#### 500 — Erreur interne

```json
{
  "error": "Internal model error."
}
```

### Exemple (cURL)

```bash
curl -X POST http://localhost:8000/predict \
 -H "Content-Type: application/json" \
 -d '{
"sepal_length_cm": 5.1,
"sepal_width_cm": 3.5,
"petal_length_cm": 1.4,
"petal_width_cm": 0.2
}'
```
