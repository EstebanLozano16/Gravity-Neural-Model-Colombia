# Colombia Trade Intelligence - Deep Learning System

Este proyecto implementa una jerarquía de redes neuronales (Keras) para analizar y predecir los flujos de exportación de Colombia entre 1995 y 2020, utilizando la teoría de gravedad comercial (PIB, Distancia y Variables Institucionales).

##  Metodología
Se construyeron 3 niveles de modelos para capturar dinámicas específicas:
1. **Universo:** Red global para todas las exportaciones.
2. **Clústeres:** Modelos especializados en Agro, Minería y Manufactura.
3. **Estratégica:** Red de nicho para productos como el Café.

##  Resultados (Benchmarking)
La red neuronal demostró ser superior a los modelos de ensamble tradicionales para capturar relaciones no lineales en el comercio:

| Modelo | Sector | MAE (Error) |
| :--- | :--- | :--- |
| **Red Neuronal** | **Manufactura** | **1.3253** (Ganador) |
| Random Forest | Manufactura | 1.3580 |
| Red Neuronal | Café | 1.5386 |
| Red Neuronal | Minería | 3.5048 |

##  Hallazgos Estratégicos
El modelo detectó una brecha de oportunidad significativa en mercados como **Ecuador y Perú** para el sector de plásticos y manufacturas, sugiriendo un potencial de crecimiento superior al 100% basado en su masa económica y cercanía geográfica.
