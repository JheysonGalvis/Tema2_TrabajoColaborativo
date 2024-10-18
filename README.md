# Tema2_TrabajoColaborativo
Carlos Andres Giraldo Saldarriaga
Fases Típicas del ciclo de vida del Aprendizaje de Máquina
1. Identificación del problema:
Definir claramente el objetivo del proyecto y determinar como el aprendizaje de máquina puede ayudar a alcanzar el objetivo.

2. Recolección de datos:
Una vez identificado el problema, la siguiente fase es la recolección de datos, estos datos serán utilizados para entrenar el modelo. Es importante asegurarse de que los datos sean relevantes, precisos y representativos del problema que se está tratando de resolver.

3. Preparación de los datos:
Los datos crudos raramente están listos para ser usados directamente para el modelo. La preparación de datos implica limpiar los datos, manejar los valores faltantes, normalizar y escalar las caracteristicas y dividir los datos en conjunto de entrenamiento y prueba.

4. Ingeniería de datos:
En esta fase, se seleccionan y entrenan modelos de aprendizaje de maquina, utilizando datos preparados. Esta etapa puede incluir la selección de algoritmos, la creacion de caracteristica (featuring engeniering), y la realización de ajustes de hiperparametros para optimizar el rendimiento del modelo.

5. Evaluación del modelo:
Una vez el modelo ha sido entrenado, es crucial evaluar su desempeño. Esto implica medir la precision, , la exactitud, el recall, la F1-score, entre otras metricas, usando el conjunto de prueba. La evaluación ayuda a determinar si el modelo es suficientemente bueno para ser desplegado.

6. Desplieque:
Desplegar el modelo significa ponerlo en producción para que pueda ser utiliado en un entorno real. Esto puede implicar la integración del modelo en aplicaciones existentes, la creación de API´s y la configuración de infraestructura para manejar las predicciones en tiempo real.

7. Mantenimiento y actualización:
El ciclo de vida de una aplicacion no termina en el despliugue. Es importante monitorear el desempeño del modelo en producción y realizar actualizaciones periodicas para mantener su precisión y relevancia. Estoa puede implicar reentrenar el modelo con nuevos datos o ajustar sus parametros.
Maria Victoria Valencia Arango

## Historia y ciclo de vida de una aplicación de Aprendizaje de Máquina

Se describirán a continuación las etapas o fases del ciclo de vida para aplicaciones de aprendizaje de máquinas.

1. Identificación del problema
2. Recolección de datos
3. Preparación de datos
4. Ingeniería de modelos
5. Evaluación del modelo
6. Despliegue
7. Mantenimiento y actualización

De acuerdo con las etapas mencionadas, se debe tener en cuenta que para iniciar a desarrollar, es importante conocer el reto a resolver, tener claridad del contexto del problema y de esta manera avanzar con las demás etapas donde se encuentran involucrados los datos: recolección, proceso de limpieza y calidad de datos para posteriormente iniciar el modelado analítico de acuerdo a la necesidad del negocio.

Adicionalmente, se requiere hacer una evaluación del modelo para evidenciar posibles mejoras y posteriormente hacer los despliegues correspondientes para los pasos a producción.

Los diferentes tipos de aprendizaje de máquina son los siguientes:

- Ciencia de datos, Matemáticas, Probabilidad estadística.
- Algoritmos de agrupación.
- Redes neuronales.
    * Aprendizaje profundo.
- Algoritmos bayesianos.
- Algoritmos de regresión.
- Árboles de decisión.


<img src="AI1.png" alt="Capas IA" width="400" height="300">

## Preparación de datos

- Análisis exploratorio de los datos: Conocer la tipología y cómo llegan los datos de la fuente.
    * Evaluación de nulos.
    * Datos faltantes.
    * Duplicidad.
    * Tipos de datos.
- Limpieza de datos: Se busca organizar la data para evitar fallos en los modelos analíticos posteriores.

Existen diferentes tipos de análisis:

- Exploratorio
- Descriptivo
- Relacional
- Explicativo
- Predictivo

<img src="tipo_datos.png" alt="tipo datos" width="400" height="300">

## Analítica de datos

Una vez se tienen los datos limpios y de calidad, se procede a tener conjuntos de entrenamiento y prueba de acuerdo a los modelos que vamos a utilizar.
Esto permitirá al modelo aprender sobre los datos y posteriormente hacer testeo sobre ellos.
Para estos conjuntos de entrenamientos se pueden tomar porciones de datos 70% para entrenamiento y 30% para testeo, ó tomar 80% para entrenamiento y 20% para testeo.

<img src="sets.png" alt="sets" width="400" height="100">


- Selección de modelo
    * Supervisado
        * Clasificación
        * Regresión
    * No supervisado
        * Agrupaciones (clustering)



Paula Jacobo Marin

HISTORIA CICLO DE VIDA DE UNA APLICACIÓN DE APRENDIZAJE DE MAQUINA

Es un modelo estructurado que sea efectivo, preciso y útil
Primero que todo debemos tener un problema o una necesidad a resolver, un proceso que donde se deben registrar métricas globales en el desempeño del modelo y compararlo con un nivel de referencia.
1.	Servidores remotos
2.	Desarrollo
3.	Producción
4.	Despliegue
5.	Monitoreo
Este último para detectar posibles fallos de software o fallas en el modelo, registrando métricas globales en el desempeño del modelo y compararlo con un nivel de referencia.
Definición del Problema: Identificar el problema específico que se quiere resolver y establecer los objetivos del proyecto.
Recopilación de Datos: Reunir datos relevantes que alimentarán el modelo. Esto puede incluir datos históricos, datos en tiempo real o datos generados artificialmente.
Preprocesamiento de Datos: Limpiar y transformar los datos para hacerlos aptos para el modelo. Esto incluye la eliminación de valores atípicos, el tratamiento de datos faltantes y la normalización.
Análisis Exploratorio de Datos: Explorar los datos para entender patrones, tendencias y relaciones. Esta etapa ayuda a informar las decisiones sobre la selección de características.
Selección y Entrenamiento del Modelo: Elegir un algoritmo de aprendizaje de máquina adecuado y entrenar el modelo utilizando los datos preprocesados. Esto implica ajustar hiperparámetros y evaluar el rendimiento del modelo.
Evaluación del Modelo: Validar el modelo utilizando métricas como precisión, Esto puede incluir la división de los datos en conjuntos de entrenamiento y prueba.
Implementación: Integrar el modelo en la aplicación o sistema donde será utilizado. Esto puede incluir la creación de una API o la incorporación en un software existente.
Monitoreo y Mantenimiento: Después de la implementación, es crucial monitorear el rendimiento del modelo en el entorno real. Esto incluye la detección de posibles sesgos, el ajuste de parámetros y la actualización del modelo con nuevos datos.
Iteración y Mejora: Basado en el monitoreo, se pueden realizar iteraciones para mejorar el modelo, volviendo a algunas de las etapas anteriores según sea necesario.
Recursos Principales necesarios para la aplicación de ML

  	DATOS DISPONIBLES


Poder de cómputo	Técnicas estadísticas
Se trata de crear modelos de observaciones y predicciones

Observaciones

•	Variables predictorias
•	Variable independiente
•	Características

Predicciones
•	Variables objetivas 
•	Variables dependientes
•	Etiquetas

TIPOS DE DATOS ESTRUCTURADOS
Datos de tipo numérico y datos categóricos
•	Atributos de los datos
•	Tipos de análisis
A partir de cierta información, determinar si ocurre o no un evento.




Daniel Calle Pulgarin
#Ciclo de vida 
El desarrollo de una aplicación de aprendizaje de máquina sigue un ciclo de vida estructurado que asegura que el modelo sea efectivo, preciso y útil. Esto se lleva a cabo por medio de unas etapas o fases :
#Machine learning 
Machine Learning representa un nuevo paradigma en la programación, donde en lugar de programar reglas explícitas en un lenguaje como Java o C ++, se entrena un sistema con datos para inferir las reglas por sí mismo.
#Identificación del problema
El primer paso en el ciclo de vida de una aplicación de aprendizaje automático es identificar el problema que se quiere resolver. Esto significa definir claramente el objetivo del proyecto y determinar de qué manera el aprendizaje automático puede ayudar a lograrlo.
#Recolección de datos 
Después de identificar el problema, el siguiente paso es recopilar datos. Estos datos se usarán para entrenar el modelo. Es crucial que los datos sean relevantes, precisos y representen bien el problema que se quiere solucionar.
#Preparación de datos
Los datos en bruto casi nunca están listos para ser utilizados directamente en el entrenamiento del modelo. La fase de preparación de datos consiste en limpiar los datos, gestionar los valores faltantes, normalizar y escalar las características, y dividirlos en conjuntos de entrenamiento y prueba.
#Ingeniería de modelos
En esta etapa, se eligen y entrenan los modelos de aprendizaje automático con los datos ya preparados. Esto incluye seleccionar los algoritmos, realizar ingeniería de características (feature engineering) y ajustar los hiperparámetros para optimizar el rendimiento del modelo.
#Evaluación del modelo
Después de entrenar el modelo, es fundamental evaluar su desempeño. Esto se hace midiendo métricas como la precisión, exactitud, recall y F1-score, utilizando el conjunto de prueba. La evaluación permite verificar si el modelo es lo suficientemente bueno para ser implementado.

#¿Cómo seleccionar el mejor modelo en un problema de machine learning?
#1. Definir el problema y las métricas de evaluación
•	Antes de seleccionar un modelo, es crucial definir el tipo de problema (clasificación, regresión, etc.) y las métricas adecuadas para evaluarlo. Las métricas comunes #incluyen:
o	Precisión (Accuracy): Qué tan bien el modelo clasifica correctamente.
o	Recall: Qué porcentaje de los casos positivos son detectados correctamente.
o	F1-Score: El balance entre precisión y recall.
o	MSE (Mean Squared Error): Usada para regresión.
o	AUC-ROC: Para medir el rendimiento en clasificación binaria.
#2. Seleccionar un conjunto de modelos candidatos
•	Elige varios modelos que puedan ser adecuados para tu problema, como:
o	Regresión lineal o logística
o	Árboles de decisión
o	Random Forest
o	Gradient Boosting Machines (GBM)
o	Redes neuronales
o	SVM (Máquinas de Soporte Vectorial)
#3. Dividir los datos en conjuntos de entrenamiento y prueba
•	Usa una parte de los datos para entrenar y otra para probar el rendimiento de los modelos. A menudo, se utiliza una división 80-20 o 70-30.
#4. Entrenar y ajustar los modelos
•	Entrena cada modelo con el conjunto de entrenamiento y realiza ajustes de hiperparámetros usando técnicas como:
o	Búsqueda en cuadrícula (Grid Search) o Búsqueda aleatoria (Random Search).
o	Validación cruzada (Cross-Validation): Divide los datos en varios subconjuntos para asegurar que el modelo generaliza bien.
#5. Comparar el rendimiento en el conjunto de prueba
•	Evalúa el rendimiento de cada modelo usando las métricas definidas y compara los resultados. El modelo con mejor rendimiento en la métrica más relevante será el mejor candidato.
#6. Evitar el sobreajuste
•	Asegúrate de que el modelo no esté sobreajustado a los datos de entrenamiento, lo que podría dar un alto rendimiento en entrenamiento pero pobre en prueba. Esto se puede verificar revisando si hay una gran diferencia entre el rendimiento en el entrenamiento y el de prueba.
#7. Interpretabilidad vs. Complejidad
•	A veces, el mejor modelo no solo es el más preciso, sino también el más fácil de interpretar. Modelos como regresión lineal o árboles de decisión son más simples de interpretar en comparación con redes neuronales profundas.
#8. Pruebas adicionales o validación externa
•	Si es posible, valida el modelo en un conjunto de datos externos que no se usaron en el entrenamiento para verificar que funcione bien en datos nuevos.
#9. Seleccionar el modelo con mejor equilibrio
•	El mejor modelo será el que tenga el mejor rendimiento en las métricas clave, que no esté sobreajustado y que sea lo suficientemente interpretable y eficiente para el propósito del proyecto.
#Despliegue 
Desplegar el modelo significa ponerlo en funcionamiento en un entorno real para que pueda ser utilizado. Esto puede requerir integrar el modelo en aplicaciones ya existentes, crear APIs para que otras aplicaciones interactúen con él, y configurar la infraestructura necesaria para gestionar las predicciones en tiempo real.
#Mantenimiento y actualización 
El ciclo de vida de una aplicación de aprendizaje automático no finaliza con el despliegue. Es crucial monitorear el rendimiento del modelo en producción y realizar actualizaciones periódicas para asegurar su precisión y relevancia. Esto puede incluir reentrenar el modelo con nuevos datos o ajustar sus parámetros según sea necesario.
*******************************************************************
*******************************************************************
#Jaime Andrés Londoño Acevedo
# Tema 2 Historia y Ciclo de Vida de una Aplicación de Aprendizaje de Máquina
## Por: Jaime Londoño

### Introduccion Machine Learning
- Programación Lineal: Entradas Datos y Reglas. Salida: Resultado
- Machine Learning: Entradas Respuestas y Datos. Salida: Reglas

#### Fases:
1 Identificación del problema: Definir el objetivo del proyecto y como el aprendizaje de máquina puede ayudar a alcanzarlo.
2 Recolección de datos: Datos relevantes, precisos y representativos del problema.
3 Preparación de datos: Lmpiar datos, manejar valores faltantes, normalizar, escalar caracteristicas, dividir en conjuntos.
4 Ingenieria de modelos: Selección de algoritmos, feature engineering, ajustes hiperparámetros. 
5 Evaluación del modelo: Medir precisión, exactitud, recall, F1-score.

   Parámetros: Variables numéricas internas que el modelo aprende
   Hiperparámetros: Variables numércias externas que nosotros debemos fijar al momento de hacer el algoritmo.

6 Despliegue: Ponerlo en producción, integrarlo con aplicaciones.

   Requerimientos de diseño: Tipo de predicción (Tiempo real o por lotes), latencia, rendimiento (numero solicit  udes por segundo)
   Alternativas de despliegue: En la nube, on the edge.

7 Mantenimiento y actualización: Monitoreo, actualizaciones periódicas, ajustar parámetros.

   Fallos de software: Código para despliegue por ejemplo
   Fallos en el modelo: Deriva de datos, Deriva de concepto.
   Monitoreo con métricas globales.
   Distribuciones estadísticas.

#### HISTORIA
- Arthur Samuel (1959), disciplina que estudia habilidad de aprendizaje de computadores.
- Tom Mitchel (1998), aprendizaje computacional.
- 1950 Artificial Intelligence, 1980 Machine Learing, 2010 Deep Learning.

Medicina: Reconocimiento de imagenes con cáncer.
Políticas publicas: predicción de riesgo (delitos, violencia, etc.)
Fiscalización: detección de fraude.
Manufactura: predicción de fallas
Agricultura: identificación de cultivos
Comercio: segmentación de clientes.

Sistemas Bioinspirados: Algoritmos genéticos, Algoritmos de enjambre, Redes Neuronales.

Aprendizaje supervisado: Clasificaciones (Arboles de decisión, Naive Bayes, XGB, Redes neuronales), Regresiones (Regresión lineal, Arbol de regresión, regresión Logística, Redes neuronales).

Aprendizaje no supervisado: Agrupaciones (Clustering).
******************************************************************
Ximena Perez Burgos

Wilmer Jamioy Tisoy 
# Tema 2 -Historia Ciclo de vida de una aplicación de Aprendizaje de máquina

Machine Learning representa un nuevo paradigma en la programación, donde en lugar de programar reglas explícitas en un lenguaje como Java o C ++, se entrena un sistema con datos para inferir las reglas por sí mismo.

## Pasos 

1. Identificación del Problema
2. Recolección de Datos
3. Preparación de Datos
4. Ingeniería de Modelos
5. Evaluación del Modelo
6. Despliegue
7. Mantenimiento y Actualización


Jhoksser Fernando Mejía Ramos

HISTORIA Y CICLO DE VIDA DE UNA APLICACIÓN DE APRENDIZAJE DE MÁQUINA 

el desarrollo de una aplicación sigue un ciclo de vida estructurado que lo hace efectivo, preciso y útil.

FASE 1. IDENTIFICACIÓN DEL PROBLEMA 
Identificar el problema que se quiere resolver, indicar claramente el objetivo del proyecto y determinar como el ML ayuda a resolver el problema
Ejemplo: predecir la demanda de un producto.

FASE 2. RECOLECCÍON DE DATOS 
Se recolectan datos para entrenar el modelo, estos datos deben ser relevantes, precisos y representativos del problema.
Ejemplo: recolectar datos hístoricos de ventas, de tendencia, busqueda en línea, datos de campañas pasadas.

FASE 3. PREPARACIÓN DE LOS DATOS
Los datos deben ser limpiados, manejar valores faltantes, normalizar y escalar características
Ejemplo: La empresa elimina datos duplicados, maneja valores faltantes, o normaliza los precios y cantidades vendidas


FASE 4. INGENIERÍA DE MODELOS.
Se selecciona el mejor modelo y se entrena con los datos ya preparados y se hacen los ajustes al modelo
Ejemplo: La empresa experimenta con varios algoritos de regresión y árboles de decisión para encontrar el mejor modelo.


FASE 5. EVALUACIÓN DEL MODELO.
en esta fase se evaluan el desempeño de los modelos y se escoje el mejor modelo segun la métrica que se quiere entrena. 
Ejemplo: La empresa usa modelos de metricas para determinar el mejor modelo para resolver su problema. Maquina de soporte vectorial, bosque aleatorio o red neuronal. usan RMSE

FASE 6. DESPLIEGUE
Significa poner el modelo en el mundo real, integrarlo a la infraestructura o aplicaciones existentes. Este despliegue puede ser en remoto o en la nube.
Ejemplo: La empresa despliega el modelo en su sistema de inventario, permitiendo predecir los pedidos o ventas.

FASE 7 MANTENIMIENTO Y ACTUALIZACIÓN
Es importante monitorear el desempeño del modelo y mantenerlo actualizado, los modelos pueden ser reentrenados para ajustar sus parametros
Ejemplo:La empresa monitorea el modelo y lo compara con las ventas reales y lo reentrena con nuevos datos para aprovechar al máximo el potencial del modelo

Victor Alfonso Gutierrez Lopez
Resumen: Historia y Ciclo de Vida de una Aplicación de Aprendizaje de Máquina
El ciclo de vida de una aplicación de aprendizaje de máquina sigue una serie de pasos estructurados para asegurar su efectividad y precisión.

Identificación del problema: Se define el objetivo que el aprendizaje de máquina puede ayudar a alcanzar. Ejemplo: predecir la demanda de productos en una tienda.

Recolección de datos: Se obtienen datos relevantes y precisos para entrenar el modelo. Ejemplo: datos históricos de ventas, tendencias de búsqueda, demografía y campañas de marketing.

Preparación de datos: Los datos se limpian, normalizan y dividen en conjuntos de entrenamiento y prueba. Ejemplo: eliminación de duplicados y manejo de valores faltantes.

Ingeniería de modelos: Se seleccionan y entrenan modelos, ajustando parámetros y creando características adicionales para optimizar resultados. Ejemplo: algoritmos de regresión y árboles de decisión.

Evaluación del modelo: Se evalúa el modelo usando métricas como el error cuadrático medio (MSE) para asegurarse de su efectividad antes de implementarlo.

Despliegue: El modelo se integra en el entorno de producción para hacer predicciones en tiempo real. Ejemplo: la predicción de demanda se usa para ajustar el inventario de una tienda.

Mantenimiento y actualización: Se monitorea y actualiza el modelo continuamente, reentrenándolo con nuevos datos para mantener su relevancia. Ejemplo: ajuste del modelo en base a datos de ventas reales.


Maritza Cristina Parra Jimenez
Historia y ciclo de vida de una aplicación de aprendizaje de máquina
Historia:  El aprendizaje automático tuvo sus inicios en las décadas de los años 1940 y 1950 con los trabajos pioneros de Alan Turing, desde entonces ha evolucionado con tecnología e innovación.
Ciclo de vida:  Este tiene varias etapas- identificación del problema, recolección y análisis de datos, desarrollo y entrenamiento del modelo, evaluación y ajuste e implementación y monitoreo continuo.

Robinson Loaiza Davila

# Historia y Ciclo de Vida de una Aplicación de Aprendizaje de Máquina

# Machine Learning
__Definicion:__ Enseñar a una computadora de la misma manera que se le enseña a un humano, 

__1. Identificación del Problema:__ Definir claramente el objetivo del proyecto y determinar cómo ML puede ayudar a alcanzar el objetivo.

__2. Recolección de los datos:__ Los datos serán utilizados para entrenar el modelo. Los datos deben ser relevantes, precisos y representativos.

__3. Preparación de los datos:__ Implica limpiar los datos, manejar los valores faltantes, normalizar, escalar las características, y dividir los datos en conjuntos de entrenamiento y prueba.

__4. Ingeniería de Modelos:__ Se entrenan modelos de aprendizaje de máquina utilizando los datos preparados. Esta etapa puede incluir la selección de algoritmos, la creación de características y la realización de ajustes de hiperparámetros para optimizar el rendimiento del modelo. (Algoritmos de regresión, y arboles de decisión).

__5. Evaluación del Modelo:__ Esto implica medir la precisión, la exactitud y otras métricas (Pruebas). Parámetros: Variables numéricas internas e Hiperparámetros son variables numéricas externas, afinación de hiperparámetros.

    •	Métricas de desempeño.
    •	Posibles arquitecturas o tipos de modelos más adecuados.
    •	Se entrena, se afina y se valida el modelo.
    •	Se elige el modelo con mejor métrica de desempeño.

__6 Despliegue:__ Llevarlo de una etapa de desarrollo a una etapa de producción para ser utilizado en un entorno real.

__7. Mantenimiento y Actualización:__  Después del despliegue es necesario monitorear el desempeño, realizar actualizaciones periódicas para mantener la precisión y relevancia.


Claudia Lorena Ramírez Franco

Machine Learning (ML): Herramienta fundamental dentro de la inteligencia artificial (IA) que permite a las máquinas aprender y mejorar a partir de datos sin necesidad de ser programadas para cada tarea específica. Su importancia ha crecido en los últimos años debido a su capacidad para resolver problemas complejos y proporcionar soluciones basadas en el análisis de grandes volúmenes de información. Este enfoque se remonta a los inicios de la IA, con el emblemático proyecto de investigación en Dartmouth en 1956.
2. Evolución del Machine Learning
•	Arthur Samuel (1959): Definió el ML como una técnica mediante la cual una máquina adquiere la capacidad de aprender de los datos sin estar programada explícitamente para realizar cada tarea.
•	Tom Mitchell (1998): Sugirió que una máquina "aprende" cuando su rendimiento en una tarea mejora gracias a la experiencia adquirida.
3. Fases del ciclo de vida de una aplicación de ML
El desarrollo de una aplicación de aprendizaje automático sigue un proceso estructurado para garantizar que el modelo sea útil y preciso. Las principales etapas del ciclo de vida son:
Definición del problema: El primer paso es identificar claramente el problema que se quiere resolver mediante el ML. Este paso incluye definir el objetivo y establecer cómo el aprendizaje automático puede aportar una solución. Ejemplo: Una tienda puede querer predecir cuáles productos serán los más vendidos en la próxima temporada para ajustar su inventario de manera eficiente.
Recolección de datos: Se recopilan los datos necesarios para entrenar el modelo. Estos datos deben ser relevantes y de buena calidad para asegurar un entrenamiento efectivo. Ejemplo: Datos históricos de ventas, tendencias de búsqueda, información demográfica y resultados de campañas de marketing.
Preparación de datos: Los datos generalmente no están listos para su uso directo, por lo que es necesario limpiarlos, gestionar valores faltantes y normalizar características importantes. Ejemplo: Una tienda podría eliminar duplicados, gestionar valores faltantes mediante imputación y normalizar los precios y cantidades vendidas.
Construcción del modelo: En esta fase se seleccionan los algoritmos adecuados, se entrenan los modelos y se ajustan los hiperparámetros. Además, se pueden crear características adicionales a partir de los datos. Ejemplo: La tienda podría probar con diferentes algoritmos de regresión o árboles de decisión para predecir la demanda de productos.
Evaluación del modelo: Se mide el rendimiento del modelo utilizando métricas específicas, como la precisión, el recall o el F1-score, y se evalúa si el modelo es adecuado para ser desplegado. Ejemplo: Utilizar métricas como el error cuadrático medio para determinar qué modelo ofrece mejores predicciones.
Despliegue del modelo: Una vez evaluado, el modelo se implementa en un entorno de producción para que sea utilizado en aplicaciones reales. Ejemplo: El modelo de predicción de demanda se integra en el sistema de gestión de inventario de una tienda.
Mantenimiento y actualización: El trabajo no termina con el despliegue. Se monitorea el rendimiento del modelo y se reentrena cuando es necesario para asegurar que siga siendo preciso y útil. Ejemplo: Si el rendimiento del modelo baja, la tienda recoge nuevos datos y reentrena el modelo para ajustar sus predicciones.
Este ciclo garantiza que las aplicaciones de aprendizaje automático se desarrollen de forma eficiente y que sigan siendo útiles y relevantes con el tiempo.
4. Problemas que puede resolver el ML
•	Clasificación: Asignar una categoría o clase a un dato, como en la detección de enfermedades.
•	Regresión: Predecir un valor numérico, como el precio de un bien o servicio.
•	Clustering: Agrupar datos en categorías sin etiquetas predefinidas, por ejemplo, en la segmentación de clientes.
5. Tipos de aprendizaje
•	Supervisado: El modelo se entrena con datos etiquetados, lo que permite hacer predicciones basadas en un conjunto de variables predefinidas.
•	Semisupervisado: Combina una pequeña cantidad de datos etiquetados con una gran cantidad de datos no etiquetados para mejorar el modelo.
•	No supervisado: El modelo encuentra patrones y estructuras en los datos sin utilizar etiquetas, lo que es útil para tareas como el agrupamiento o la detección de anomalías.
6. Herramientas y recursos relevantes
Hoy en día, existen muchas herramientas que facilitan el desarrollo de aplicaciones de aprendizaje automático. Modelos como GPT-4, DALL-E y herramientas como AutoGPT y Stable Diffusion son algunas de las más avanzadas para la generación de contenido y predicciones en tiempo real.



Roberto Alejandro Sánchez

## Analítica de datos

Una vez se tienen los datos limpios y de calidad, se procede a tener conjuntos de entrenamiento y prueba de acuerdo a los modelos que vamos a utilizar.
Esto permitirá al modelo aprender sobre los datos y posteriormente hacer testeo sobre ellos.
Para estos conjuntos de entrenamientos se pueden tomar porciones de datos 70% para entrenamiento y 30% para testeo, ó tomar 80% para entrenamiento y 20% para testeo.

Claudia Cardenas
1. Identificar el Problema 
Determinar el objetivo
2. Recolectar Datos
Relevantes, precisos, representativos
3. Preparacion de Datos
Limpiarlos, completar faltantes…
4. Ingenieria de modelos
Seleccionar algoritmos, determinar características y realizar ajustes de hiperparametros
5. Evaluar modelos
Mirar precision y exactitud
6. Despliegue o Implementacion
Ponerlo a producir = pasar del dllo a la implementacion
7. mantenimiento y actualizacion
Corregir errores y fallas del modelo y/o actualizarlo.




Luis Angel Montoya Suárez
## Ciclo de vida de una aplicación de  de aprendizaje de máquina

1. Identificación de un problema
2. Recoración de datos
3.  Reparación de datos
4.  Ingeniería de modelos. 
5.  Evaluación del modelo
6. Despliegue
7. Mantenimiento de actualización

          ## Diagrama procesos Machine Learning

1. Recolección de datos
2. Exploración de datos
3. Reprocesamiento
4. Entrenamiento del modelo
5. Evaluación del modelo
6. Refinamiento del modelo
7. Despliegue de modelo

   TIPOS DE BASES DATOS

Yessica Marcela Triana Cordoba
Inteligencia Articaal
Tipos aprendage IA
Mision 1:> Histona ado de vida do una aplicacion • Superuzadas =)
Aprendizaje de maquina
Machinee learning -> datos • reglas sobro datos, darle dados al programa y la maquiina diga las reglar
(Pythons Idioma programación
J. Identificar el problema
2. Recolectar datos.
3. Preparar datos →→ Cual sirve y cual no
4.Ingenieria de modelos : selección de lgoritmos  y creación de caracteristicas
5. Evaluar modelos: medir precisión, exactitus, recall FI socre
Parametros de un modelo. Variable nuero internas de aprendizaje
Has tiquetados No sopercicado => datos no etiquetados Reforzados aprendo toma doxción
a. Ingenieria de Modelos → apton maquino = Seleccion algoritmos 15. Cual modelo + Medor
RMSE = Valor real Predisiones - elevadas al cuadrado – sumarlas y divididas entro la suma real de datos a comparar
lternativass Nube -Rest APY desplegue Ondly Edge
Desarrollo Prototipo modelo
6. Despleger→ Producirlo para que pueda ser usado
 Integrar modelos con aplicaciones inesistentes Creacion APLs
Creacion APIS
Configuración de información – Predición real

Luis Fernando Meneses Caviedes

<img src="Phyton1_img.jpg" alt="Capas IA" width="400" height="218">

Tema 2 - Historia y Ciclo de Vida de una Aplicación de Aprendizaje de Máquina

El desarrollo de una aplicación de aprendizaje de máquina sigue un ciclo de vida estructurado que asegura que el modelo sea efectivo, preciso y útil. Cada etapa de este ciclo es crucial para el éxito del proyecto. A continuación, se describen las fases típicas en el ciclo de vida de una aplicación de aprendizaje de máquina, junto con ejemplos que ilustran cada paso.

Introducción a Machine Learning (ML Zero to Hero, parte 1)
Machine Learning representa un nuevo paradigma en la programación, donde en lugar de programar reglas explícitas en un lenguaje como Java o C ++, se entrena un sistema con datos para inferir las reglas por sí mismo. Pero ¿cómo es en realidad el ML?
VIEW ON YouTube

El Machine Learning es el camino a la IA, aquí aplicaremos el lenguaje Phyton, cuando se trata de programación tradición se aplican Reglas para procesar los datos y asi obtener un resultado, en Machine Learning invertimos los papeles y en lugar de brindarle las reglas le damos los resultados o los datos para obtener las reglas.


Sebastian de Jesus Garcia Lopez
las fases de un ciclo de vida de una aplicacion de aprendizaje de maquina son:

* Identificación del problema: Se define el objetivo del proyecto, determinando cómo el ML puede solucionar el problema identificado. Por ejemplo, una empresa podría querer predecir la demanda de productos para optimizar su inventario.

* Recolección de datos: Se recopilan datos relevantes que alimentarán el modelo, asegurando que sean representativos y de calidad. En nuestro ejemplo, la empresa recogería datos de ventas históricas, demografía y campañas de marketing.

* Preparación de datos: Los datos crudos se limpian y preparan para el entrenamiento. Esto incluye eliminar duplicados, manejar valores faltantes, normalizar características y dividir los datos en conjuntos de entrenamiento y prueba.

* Ingeniería de modelos: Se seleccionan y entrenan los modelos utilizando algoritmos adecuados. También se ajustan los hiperparámetros y se realizan mejoras en la representación de las características.

* Evaluación del modelo: El rendimiento del modelo se evalúa usando diversas métricas, como la precisión, exactitud y F1-score. Esto permite seleccionar el mejor modelo para resolver el problema.

* Despliegue: El modelo se integra en sistemas productivos, como una API o un sistema de gestión de inventario en tiempo real.

* Mantenimiento y actualización: El ciclo continúa después del despliegue. Se monitorea el rendimiento del modelo y, si es necesario, se actualiza o reentrena con nuevos datos.

Este ciclo de vida permite que los sistemas de ML se mantengan eficientes y alineados con los objetivos del negocio a lo largo del tiempo.

En resumen el ciclo de vida de una aplicación de aprendizaje de máquina comienza con la identificación del problema, donde se define el objetivo del proyecto. Luego se realiza la recolección de datos relevantes para alimentar el modelo. Después, los datos se preparan para el modelo mediante la limpieza y normalización. En la fase de ingeniería de modelos, se seleccionan algoritmos y se entrenan modelos que luego se evalúan. Una vez seleccionado el mejor modelo, se despliega en producción, y finalmente se realiza un mantenimiento continuo para asegurar su relevancia y precisión.



Valentina Cepeda Duque

Juan Diego Araque Muñoz


José Luis Cardeño Tejada

Se habla sobre el cilco de vida de una aplicación de aprendizaje de máquina.
Así el machene learning, sería el cómo hacer que la computadora aprenda como un ser humano.
Hay una reglas en los que se aplican unos datos que finalmente generan respuestas o resultados.
Ahora se invertiria el aprendizaje dandole las respuestas en forma de datos a la computadora y hacemos que la computadora haga las reglas


Fases de ciclo de vida:

1. Identificación del problema: ----> Objetivo del proyecto y dererminar como el aprendizaje de la máquina puede ayudar a alcanzar este objetivo.

2. Recolección de datos: con ellos se entrenará el modelo.
Los datos debe ser relevantes y precisos del problema que se está tratando resolver.

3. Preparación de datos: implica limpiar los datos, manejas los valores faltantes, normalizar y escalar las características
y dividir los datos en conjuntos de entrenamiento y prueba.
Ej: limpiar datos duplicados, manejo valores faltantes, normalizae caracteristicas como precios y cantidades, etc.

4. Ingeniería de modelos: selección y entrenamiento de modelos de aprendizaje. En esta etapa puede incluir la selección de algoritmos, 
ajustes de hiperparamtros

5. Evaluación del modelo: después de entrenada la IA, hay que evaluar su desempeño. Medir la precisiónexcatitud, el recall, etc.
LA idea es seleccionar el modelo más adecuado (las mejores predicciones). Se debe entonces entrenar mulrtiples modelos y evaluar
su desempeño utilizando el set de datos y elegir el mejor modelo que genere las mejores predicciones (establecer la metrica de 
desempeño) es una formula matemática. RMSE Raiz cuadrada del error cuadratico medio (entre más pequeño ese valor, mejor)

6. Despliegue: ponerlo en producción para ser utilizado en el entorno real (Mlops)

7. Mantenimiento y actualización. Para brindar soporte constante a la IA

Geovanny Vergara Ramírez 
****
Historia y ciclo de vida de una aplicación de Aprendizaje de Máquina

El desarrollo de una aplicación de Aprendizaje de Máquina (Machine Learning) sigue un ciclo de vida compuesto por siete etapas principales:

1. Identificación del problema
2. Recolección de datos
3. Preparación de datos
4. Ingeniería de modelos
5. Evaluación del modelo
6. Despliegue
7. Mantenimiento y actualización

<img src="ImagenGeo.svg" alt="sets" width="366" height="230">

Para iniciar el desarrollo, es crucial comprender claramente el desafío a resolver y su contexto. Esto permite avanzar eficientemente en las etapas subsiguientes, que involucran el manejo de datos: recolección, limpieza y aseguramiento de calidad. Posteriormente, se procede al modelado analítico según las necesidades específicas del negocio.

Es esencial evaluar el modelo para identificar posibles mejoras antes de realizar los despliegues correspondientes en producción.

Tipos de Aprendizaje de Máquina:
- Ciencia de datos, Matemáticas, Probabilidad estadística
- Algoritmos de agrupación
- Redes neuronales (incluyendo Aprendizaje profundo)
- Algoritmos bayesianos
- Algoritmos de regresión
- Árboles de decisión

Preparación de datos:
1. Análisis exploratorio:
   - Evaluación de valores nulos
   - Identificación de datos faltantes
   - Detección de duplicados
   - Verificación de tipos de datos

2. Limpieza de datos: Organización para prevenir errores en modelos analíticos futuros

Tipos de análisis:
- Exploratorio
- Descriptivo
- Relacional
- Explicativo
- Predictivo

Analítica de datos:
Una vez que los datos están limpios y son de calidad, se crean conjuntos de entrenamiento y prueba. Comúnmente, se utilizan proporciones de 70/30 o 80/20 para entrenamiento y prueba, respectivamente.

Selección de modelo:
1. Supervisado:
   - Clasificación
   - Regresión
2. No supervisado:
   - Agrupaciones (clustering)

Este enfoque estructurado garantiza un desarrollo eficaz y eficiente de aplicaciones de Aprendizaje de Máquina, desde la conceptualización hasta la implementación y mantenimiento.

****

Carlos Perea

# Historia

El aprendizaje de máquina (ML) tiene sus raíces en la década de 1950, cuando se desarrollaron los primeros algoritmos para la inteligencia artificial (IA). Con el avance de la computación, en los años 80 y 90, surgieron técnicas como las redes neuronales. Sin embargo, fue en la última década, gracias a la disponibilidad de grandes volúmenes de datos y potentes capacidades de procesamiento, que el aprendizaje de máquina ha alcanzado un auge significativo. Esto ha llevado a su aplicación en diversos campos, como la visión por computadora, el procesamiento del lenguaje natural y la automatización de procesos.

## Ciclo de Vida de una Aplicación de Aprendizaje de Máquina
1. Definición del Problema: Identificar el problema que se quiere resolver y establecer objetivos claros.

2. Recolección de Datos: Recopilar datos relevantes y de calidad, que son fundamentales para entrenar el modelo.

3. Preprocesamiento de Datos: Limpiar y transformar los datos, eliminando ruidos y tratando datos faltantes.

4. Selección del Modelo: Elegir el algoritmo adecuado que mejor se adapte al problema (regresión, clasificación, clustering, etc.).

5. Entrenamiento del Modelo: Utilizar los datos preprocesados para entrenar el modelo, ajustando sus parámetros.

6. Evaluación del Modelo: Probar el modelo con un conjunto de datos no visto para medir su rendimiento y precisión.

7. Optimización: Refinar el modelo ajustando hiperparámetros y mejorando la calidad de los datos.

8. Despliegue: Implementar el modelo en un entorno de producción, asegurando que pueda integrarse con otras aplicaciones.

9. Mantenimiento y Monitoreo: Supervisar el rendimiento del modelo en el tiempo, ajustándolo según sea necesario para adaptarse a nuevos datos o cambios en el entorno.

Este ciclo es iterativo, y cada fase puede requerir revisiones y ajustes para mejorar continuamente el rendimiento de la aplicación.


### MLOps, o "Machine Learning Operations", es un conjunto de prácticas y herramientas que busca integrar el desarrollo y la operación de modelos de aprendizaje de máquina en un flujo de trabajo coherente y eficiente. Aquí te detallo sus funciones principales:

1. Colaboración Interdisciplinaria: Facilita la comunicación y colaboración entre equipos de data scientists, ingenieros de datos y operaciones, asegurando que todos trabajen hacia objetivos comunes.

2. Automatización del Ciclo de Vida del Modelo: Automatiza etapas del ciclo de vida del aprendizaje de máquina, como el entrenamiento, la validación y el despliegue de modelos, lo que aumenta la eficiencia y reduce errores.

3. Gestión de Datos: Ayuda en la recopilación, limpieza y manejo de grandes volúmenes de datos necesarios para entrenar modelos, garantizando su calidad y accesibilidad.

4. Versionado de Modelos: Permite el seguimiento de versiones de modelos y datasets, lo que facilita la reproducibilidad y el control de cambios.

5. Monitoreo y Mantenimiento: Supervisa el rendimiento de los modelos en producción, detectando desviaciones o deterioro del rendimiento, y gestionando la actualización o reentrenamiento de modelos según sea necesario.

6. Escalabilidad: Asegura que los modelos puedan escalar en producción, optimizando recursos y garantizando tiempos de respuesta adecuados.

7. Cumplimiento y Seguridad: Implementa prácticas que aseguran el cumplimiento normativo y la seguridad de los datos, lo cual es crucial en muchos sectores.

MLOps busca optimizar el proceso de implementación y mantenimiento de modelos de aprendizaje de máquina, garantizando que sean efectivos y sostenibles a lo largo del tiempo

Juan Pablo Guzmán Moreno

###1. Entender el problema
Primero, tienes que saber bien cuál es el problema que quieres resolver con ia. ¿Qué necesitas mejorar o predecir usando los datos?

###2. Reunir y preparar los datos
Luego, debes buscar los datos necesarios para resolver el problema. Puede ser información de clientes, ventas, imágenes, etc.
Después, es importante limpiar esos datos: quitar errores, completar datos que faltan y organizarlos para que el modelo los entienda bien.

###3. Dividir los datos
Separas los datos en dos grupos: uno para entrenar el modelo y otro para probar qué tan bien funciona.
A veces también divides los datos en más partes para probar varias veces que el modelo funciona bien en diferentes escenarios.

###4. Elegir el modelo
Ahora, eliges qué tipo de modelo usar (hay muchos tipos, como los que predicen números o clasifican cosas en categorías).
Luego, entrenas el modelo con los datos para que "aprenda" a hacer predicciones.

###5. Evaluar el modelo
Cuando el modelo ya ha aprendido, lo pruebas para ver qué tan bien predice.
Si no lo hace bien, ajustas algunas configuraciones para mejorar los resultados.

###6. Ponerlo a funcionar
Una vez que el modelo está listo, lo integras en una aplicación o sistema real, para que pueda hacer predicciones en el día a día.
También necesitas vigilarlo para asegurarte de que sigue funcionando bien.

###7. Mantener y mejorar
Con el tiempo, los datos pueden cambiar o el modelo puede dejar de ser tan efectivo, así que es importante actualizarlo y entrenarlo de nuevo con datos más recientes.
Siempre puedes seguir mejorando el modelo para que sea más preciso.
Este proceso es cíclico, lo que significa que siempre puedes volver a pasos anteriores para mejorar el modelo o adaptarlo a nuevas necesidades.

Maria Camila Castro Isa
*El desarrollo de una aplicación de aprendizaje de máquina sigue un ciclo de vida estructurado que asegura que el modelo sea preciso, efectivo y útil. Cada etapa de este ciclo es crucial para el éxito del proyecto.
##Fases en el ciclo de vida de una aplicación de aprendizaje de máquina:
##1. Identificación del problema:
Se requiere responder, esto implica definir el objetivo del proyecto y determinar cómo el aprendizaje de máquina puede ayudar a alcanzar este objetivo.
##2. Recolección de datos:
Que serán utilizados para entrenar el modelo. Deben ser relevantes, precisos y representativos del problema que se está tratando de resolver.
##3. Preparación de datos:
Implica limpiar los datos, manejar los valores faltantes, normalizar y escalar las características, y dividir los datos en conjuntos de entrenamiento y prueba.
##4.Ingeniería de modelos:
Donde se seleccionan y entrenan modelos de aprendizaje de máquina, utilizando la prueba y preparación de modelos. Incluye la creación de características (feature engineering) y la realización de ajustes de hiperparámetros para obtener un mejor rendimiento del modelo. Ver video de introducción a machine learning.
##5. Evaluación del modelo:
Esto implica medir la precisión, la exactitud, el recall, la F-score, entre otras métricas. Usando el conjunto de prueba, la evaluación ayuda a determinar si el modelo es suficientemente bueno para ser desplegado en un entorno real. Ver video de cómo seleccionar el mejor modelo para un problema en específico.
##6. Despliegue:
Consiste en ponerlo en producción para que pueda ser utilizado en el entorno real. Esto puede implicar la estructura del modelo en aplicaciones clientes, la creación de APIs, el configurado de infraestructura para manejar las predicciones en tiempo real.
##7. Mantenimiento y actualización:
Es importante monitorear el desempeño del modelo en producción y realizar actualizaciones periódicas para mejorar la precisión y relevancia. Esto puede implicar reentrenar el modelo con nuevos datos o ajustar los parámetros. (Ver video sobre el monitoreo en el ML.)

Diana Carolina Arias Valencia

Edisson Ferley Echavarria Marin

# Ciclo de vida de una app

etapas estructuradas que aseguran efectividad, precision y utilidad de una app

## fases tipicas

### identificaion del problema
	Implica objetivo del proyecto y como es relevante el aprendizaje de maquina para la resolucion

### Recoleccion de datos
	para entrenar modelo, deben ser relevantes, precisos y representativos

### Preparacion de datos
	filtrar datos de valores faltantes, duplicados, escalar, nulos, rellenar; dividir en datos de entrenamiento y prueba

### Ingenieria de modelos
	seleccion y entrenamiento de modelos con datos preparados. Modelos: Regresion, arboles de decision

### Evaluacion del modelo
	evaluar desempeño de modelo entrenado: precision(MSE error cuadratico medio, r-cuadrada), exactitud, recall, f1-score. Determina si modelo es suficientemente bueno. 

### Mejores predicciones
		parametros: variables nuemericas internas que el modelo aprende el algoritmo de entrenamiento
		hiperparametros: variable numericos extenas que se ajustan cuando se crea algoritmo
		afinacion de hiperparametros: encontrar mejores valores para generar mejores predicciones
		set de datos entrenamiento, validadcion y pruebas
		validacion cruzada
  
### Despliegue
	poner modelo en produccion, implica integracion a modelos existentes

### Mtto y actualizacion
	monitoreo de desempeño, actualizacion periodicas para precision y relevancia, puede implicar reentrenar con nuevos datos o ajustar parametros

Diana Maribel Balaguera Arroyave

Sebastian Castañeda Garcia

Machine Learning: Enseñale a una computadora a aprender de la misma forma en que aprende un humano
en la programación tradicional se le dan la reglas a una computadora para que encuentre las respuestas en cambio en el machine learning se le dan las respuestas a la computadora y que sea la que encuentre las reglas. en un programa de machine learning, se obtiene un conjunto de datos con patrones inherentes y la computadora aprende cuales son esos patrones.

CICLO DE VIDA DE UNA APLICACION DE APRENDIZAJE DE MAQUINA

1 Identificación del problema
2 recolección de datos, que sean relevantes, precisos y representativos del problema.
3 Prepacion de datos; limpiándolos, manejando valores faltantes, normalizando y escalando características, y dividir los datos en conjuntos de entrenamiento y prueba.
4 Ingeniería de modelos, se selecciónan y entrenan modelos de aprendizaje de maquina utilizando los datos preparados.


Carlos Bolaños
IDENTIFICAR EL PROBLEMA
RECOLECTAR DATOS
PREPARACIÓN DE DATOS
INGENIERIA DE MODELOS
EVALUACIÓN DEL MODELO
DESPLIEGUE DEL MODELO
MANTENIMIENTO Y ACTUALIZACIÓN


Aurelio Cheveroni
El texto del resumen.

# Tema 2 Historia Ciclo de vida de una aplicación de Aprendizaje de Máquina (AM)
## Resumen 
### Video: Introduccion a Machine Learning
- El cerebro humano reconoce lo que esta viendo, la inteligencia artificial limita el reconocimiento a los datos que tenga
- Reglas + Datos --PrgramacionTradicion--> Resultados
- Machine Learning: Dar respuesta a programa, y que reconozca en un conjunto de datos aquellos que tengas esas caracteristicas
- Funcion de perdida y optimizados (mejora la aproximacion) --> Parametros clave para ML

### Fases tipicas en el ciclo de vida de una aplicación de AM
1. **Identificacion del Problema**
Definir el objetivo del proyecto y determinar el AM que puede ayudar a darle solucion.

2. **Recolección de Datos**
Obtencion de datos *relevantes, precisos y representativos*.

3. **Preparación de Datos**
Limpiar datos, caracterizar valores faltantes, normalizar y escalar las caracteristicas, y dividir los datos en conjuntos de entrenamiento y prueba.

4. **Ingenieria de Modelos**
Seleccion y entranamiento de modelos de AM. Incluye seleccion de algoritmos, creacion de caracteristicas y realizacion de ajustes de hiperparámetros.

5. **Evaluación del Modelo**
Desempeño y evaluacion piloto en función de la precisión, la exactitud, el recall, la F1-score, entre otras métricas.

### Video: ¿Cómo seleccionar el MEJOR MODELO en un problema de Machine Learning?
- Mejor modelo = Mejores predicciones
- Parametros: Variables numericas internas que el algoritmo aprende
- Hiperparámetros: Variables numericas externas, se dijan al programar el algoritmo.
- Afinacion de hiperparametro: Seleccion de mejores valores hiperparametros
- Para evaluar el modelos, se pude hacer:
    - Sets de entrenamiento, validacion y prueba.
    - Validacion cruzada y *k-fold cross-validation*.
- Prediccion: Calculo de valor numerico, apartir de datos numericos de entrada.
- Metrica de Desempeño: Formula matematica, que permite estimar la eficiencia del modelo. Eje: RMSE: Root Mean Square Error
![alt text](RMSE.JPG)
- Entre menor sea el valor de RMSE, mejores seran las predicciones
- **Modelos de ML:** Maquina de Soporte Vectorial, Bosque Aleatorio, Red Neuronal
- Aquel modelo que obtenga un RMSE menor, sera mejor la prediccion.
- El modelo tambien depende de la capacidad del equipo que corre el programa de ML.

6. **Despliegue**
Poner a trabajar el modelo en un entorno real.

### Video: El DESPLIEGUE en el Machine Learning (MLOps)
- Machine Learning Enginnering: Conjunt de practicas, que busca lograr el desplieuge de modelos de ML de forma *eficiente*.
- Requerimientos para elegir el tipo de despliegue:
    - Tipo de prediccion: En tiempo real (prediccion inmediata) o por lotes (prediccion no inmediata)
    - Latencia: Tiempo de respuesta requerido, desde la introduccion de la informacion y la prediccion
    - Rendimiento: Numero de solicitudes por segundo que pude aguantar el modelo
    - Complejidad del modelo: Capacidad de Maquina
    - Despliegue en la nube: Predicciones a traves de internet. Modelos complejos y de alta latencia.
    - Despliegue on the edge: En el dispositivo, modelos poco complejos y de baja latencia. (dispositivos moviles).
- Herramientas de despligue mas usadas: 
    - **Locales:** No permiten llevar el modelo a produccion. Se despliegan localmente
        - FlaskAPI: Empaqueta modelo como API, para ingresar desde nuestro navegador.
        - TensorFlow o TorchServe: Pocas lineas de codigo, permiten desplegar modelos localmente.
        - On the edge
    - **Out of the box:** 
        - StreamLite: Se aloja el codigo en un servidor, con ciertas restricciones de memoria y computo
    - **En la nube**
        - El computo se hace en servidores remotos. Eje: AWS, Google Could, etc.
        - Incluye todo el servicio de ML Operations


7. **Mantenimiento y actualizacion**
Monitoreo del desempeño del modelo en producción y realizar actualizaciones periódicas para mantener su precisión y relevancia. 

### Video: EL MONITOREO en el Machine Learning
- Machine Learning Operations!
- Desplegar el modelo no es la ultima fase del proceso, porque puede sufrir una degradacion en el desempeño
- Los fallos en el desempeño pueden ser por: Fallos de Software (factores externos) y Fallos del Modelo
    - Fallos del modelo: Los mas comunes son "Variacion en la distribucion". Deriva de datos y deriva de conceptos.
- Tipos de Monitoreo
    - Monitoreo con metricas globales: No permite ver ls razones de fondo que dan esa degradacion.
    - Monitoreo a traves de sitribuciones estadisticas: Periodicamente calcular pruebas estadisticas, y ver diferencias significativas, si las hay, habria deriva de datos. Si es razonablemente gradne, seria Deriva de Cooncepto.


 # Miguel Angel Valencia Ortiz

 *********************************************************************************************



Javier Eduardo Quintero Maken

# Resumen del Ciclo de Vida de una Aplicación de Aprendizaje de Máquina
## Identificación del Problema: Definir claramente el problema a resolver.
Ejemplo: Predecir la demanda de productos en una tienda.
## Recolección de Datos: Reunir datos relevantes y precisos para entrenar el modelo.
Ejemplo: Datos históricos de ventas y tendencias de búsqueda.
## Preparación de Datos: Limpiar y normalizar los datos, dividiéndolos en conjuntos de entrenamiento y prueba.
Ejemplo: Eliminar duplicados y manejar valores faltantes.
## Ingeniería de Modelos: Seleccionar y entrenar diferentes modelos de aprendizaje de máquina.
Ejemplo: Experimentar con algoritmos de regresión y árboles de decisión.
## Evaluación del Modelo Medir el desempeño del modelo utilizando métricas adecuadas. (métrica de desempeño) la métrica se escogerá dependiendo del tipo de problema que queramos resolver. (RMSE)
### Seleccionar los posibles modelos a entrenar (algoritmos de machine learning canal YouTube)
1.Máquina de soporte vectorial
2.Bosque aleatorio
3.Red neuronal
Entrenar, afinar y validar cada modelo 
Tomar cada uno de los modelos encontrar sus parámetros y encontrar el set de los hiperparametros (mejores predicciones), luego validarlo calcular su RSME podemos utilizar los set de entrenamiento, validación y prueba o la validación cruzada y al final tendremos un número que será el RSME de cada modelo y que indicara que tan buenas predicciones esta generando cada uno de estos modelos con datos que nunca había visto 
Máquina de soporte vectorial 
Seleccionar el mejor modelo 
Estamos hablando del modelo que genere las mejores predicciones posible, sera el mejor modelo el que contenga el menor RMSE
# Despliegue Implementar el modelo en producción para su uso real.
Ejemplo: Integrar el modelo en el sistema de gestión de inventario.
# Mantenimiento y Actualización: Monitorear y actualizar el modelo para mantener su efectividad.
Ejemplo: Reentrenar el modelo con nuevos datos si es necesario.
Este ciclo de vida asegura que las aplicaciones de aprendizaje de máquina se desarrollen de manera efectiva y sean útiles en el contexto real.
## Conclusión
Este ciclo de vida asegura que las aplicaciones de aprendizaje de máquina se desarrollen de manera sistemática, permitiendo a las organizaciones aprovechar al máximo el potencial de esta tecnología. Cada etapa es crucial para el éxito final del proyecto, asegurando que el modelo no solo sea preciso, sino también útil en el contexto real de negocio.

Francia Elena Loaiza Guevara
Introducción al Machine Learning (ML)
El Machine Learning (ML) es una disciplina clave dentro de la Inteligencia Artificial (IA) que se centra en permitir que las máquinas aprendan a partir de los datos. Este aprendizaje permite hacer predicciones, clasificaciones y tomar decisiones basadas en patrones ocultos en los datos. En esencia, el ML busca que las máquinas sean capaces de realizar tareas de manera similar a los humanos, sin necesidad de estar programadas explícitamente para cada acción.
El campo del ML ha evolucionado considerablemente desde sus inicios en los años 50 y 60, cuando surgieron los primeros intentos por crear máquinas que pudieran aprender y resolver problemas. Actualmente, gracias al poder del procesamiento de datos y los avances en algoritmos, el ML se utiliza en diversas aplicaciones que van desde la medicina hasta la industria y la agricultura.
Tipos de Problemas Abordados con ML
Existen tres tipos principales de problemas que el ML puede resolver:
1.	Clasificación: El objetivo es asignar etiquetas a los datos. Esto incluye la clasificación de correos electrónicos como spam o no spam, o la identificación de enfermedades a partir de imágenes médicas.
2.	Regresión: En este caso, el ML predice un valor numérico continuo, como predecir el precio de una casa basado en datos históricos de ventas o proyectar las ventas futuras de un producto.
3.	Clustering: El objetivo es agrupar datos que no están etiquetados en categorías, basándose en características similares. Por ejemplo, la segmentación de clientes para campañas de marketing.
Aprendizaje Supervisado, No Supervisado y Semisupervisado
El aprendizaje supervisado es el más común y consiste en entrenar al modelo con datos etiquetados, es decir, datos que contienen la respuesta correcta. Por ejemplo, clasifica imágenes de gatos y perros, el modelo es entrenado con imágenes ya etiquetadas como "gato" o "perro".
El aprendizaje no supervisado no cuenta con datos etiquetados, y el modelo debe buscar patrones o relaciones en los datos sin guía. Ejemplo es el clustering, donde el modelo agrupa datos similares sin saber a qué grupo pertenecen.
El aprendizaje semisupervisado combina ambos enfoques, utilizando una pequeña cantidad de datos etiquetados junto con una gran cantidad de datos no etiquetados para entrenar al modelo.
Aplicaciones Reales del ML
El Machine Learning ya está teniendo un impacto significativo en múltiples industrias y sectores. Ejemplo:
•	Medicina: Los algoritmos de ML se utilizan para diagnosticar enfermedades, como el cáncer, analizando imágenes médicas.
•	Administración Pública: Se emplea en la predicción de riesgos sociales, como la probabilidad de delitos o la identificación de fraudes en sistemas fiscales.
•	Agricultura: Permite a los agricultores identificar cultivos y predecir la calidad de las cosechas mediante análisis de imágenes y datos meteorológicos.
•	Manufactura: Los sistemas de ML ayuda a detectar fallos en maquinaria antes de que ocurran, minimizando el tiempo de inactividad mediante el mantenimiento predictivo.
Proceso de Aplicación del ML
El éxito de un proyecto de ML depende en gran medida de la calidad del proceso de entrenamiento y validación del modelo. Este proceso incluye dividir los datos en dos conjuntos: uno para entrenar el modelo (training set) y otro para probar su precisión (test set).
Una práctica común es reservar el 70% de los datos para el entrenamiento y el 30% para las pruebas. La clave aquí es que los datos de prueba deben ser nuevos para el modelo, lo que permite evaluar su capacidad para generalizar y predecir correctamente en situaciones reales.
El ajuste del modelo se llama calibración, y es esencial para evitar problemas como el "overfitting", donde el modelo se ajusta demasiado a los datos de entrenamiento y no funciona bien con datos nuevos.
Retos y Futuro del Machine Learning
El futuro del ML es prometedor, con desarrollos como los modelos de lenguaje generativo (GPT-4, LaMDA) que ya están revolucionando cómo interactuamos con las máquinas. Sin embargo, aún hay desafíos importantes. A pesar de los avances, las máquinas todavía carecen de sentido común y la ética en el ML es un tema crucial, especialmente en términos de privacidad y uso responsable de los datos.
Los desarrollos en Inteligencia Artificial General (AGI) —máquinas que puedan aprender cualquier tarea humana— aún están en fases iniciales. Aunque se han logrado grandes avances, las máquinas todavía están lejos de igualar la complejidad del cerebro humano en cuanto a flexibilidad, adaptabilidad y juicio moral.

Conclusión
El Machine Learning transforma muchas áreas de nuestra vida cotidiana y tiene el potencial de seguir cambiando la forma en que abordamos problemas complejos en la ciencia, la tecnología y la innovación. Desde diagnósticos médicos hasta la optimización de procesos industriales, el ML nos permite ir más allá de lo que antes era posible, aunque también nos exige ser más responsables en su uso y conscientes de sus limitaciones y riesgos.


