# Monitoreo Conceptual del Modelo

## Objetivo

Monitorear el modelo de predicción de churn después del despliegue para detectar degradación, cambios en los datos y necesidad de reentrenamiento.

## Métricas a monitorear

### 1. Métricas del modelo
- Accuracy
- Precision
- Recall
- F1-score

### 2. Métricas operativas
- Cantidad de predicciones
- Tiempo de respuesta del endpoint
- Errores de la API
- Disponibilidad del servicio

### 3. Drift de datos
Comparar la distribución de variables actuales contra las usadas en entrenamiento:

- tenure_months
- monthly_charges
- support_tickets_last_6m
- contract_type
- internet_service

### 4. Drift del modelo
Revisar si el desempeño baja cuando se comparan predicciones contra datos reales posteriores.

## Acciones correctivas

Si se detecta degradación:

1. Analizar nuevas distribuciones de datos.
2. Ejecutar nuevo preprocesamiento.
3. Reentrenar el modelo.
4. Comparar métricas en MLflow.
5. Aprobar nueva versión.
6. Desplegar el nuevo modelo.

## Flujo conceptual


API Flask
   |
Predicciones
   |
Logs y métricas
   |
Evaluación de drift
   |
Reentrenamiento
   |
Nuevo modelo versionado