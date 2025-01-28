Proyecto de Gemelo Digital para Sensores 2D
Autora: Valeria Solarte
¿Qué es y para qué es?
Este proyecto implementa un modelo de gemelo digital utilizando redes neuronales para sensores bidimensionales (y = f(x)).
El propósito es predecir y modelar la relación entre el tiempo y la temperatura de un sensor, y permitir simulaciones tanto en la dirección directa (y = f(x)) como en la inversa (x = f⁻¹(y)).

¿Cómo lo hago funcionar?
Prerrequisitos:
Tener Python instalado en el sistema (versión 3.7 o superior recomendada).
Instalar las siguientes bibliotecas necesarias:
bash
Copiar
Editar
pip install numpy pandas matplotlib seaborn statsmodels keras plotly
Tener el archivo de datos DatosFinalFinal.xlsx con las mediciones de tiempo y temperatura.
Acceso a Google Drive configurado con autenticación para cargar los datos.
Ejecución:
Cargar los datos:

El programa descarga el archivo DatosFinalFinal.xlsx desde Google Drive y lo utiliza como fuente de datos para entrenar el modelo.
Entrenamiento del modelo:

Ejecute el script principal para entrenar los modelos y = f(x) y x = f⁻¹(y) con funciones de activación configurables.
Simulaciones:

Para realizar predicciones:
Utilice el modelo directo para predecir temperatura en función del tiempo: funXY(x).
Utilice el modelo inverso para estimar tiempo en función de la temperatura: invfunYX(y).
Guardar modelos:

Los modelos entrenados se pueden guardar en archivos .h5 para su uso posterior.
Ejemplo de ejecución:
bash
Copiar
Editar
python NNDigitalTwinTemperature.py
¿Cómo está hecho?
El proyecto está implementado en Python utilizando Keras y otros paquetes para análisis y visualización. La estructura incluye:

Componentes:
XNNSensor:

Clase principal que gestiona los modelos de gemelo digital directo (y = f(x)) e inverso (x = f⁻¹(y)).
Métodos:
getfunXYModel: Entrena el modelo para y = f(x).
getinvfunYXModel: Entrena el modelo inverso x = f⁻¹(y).
saveModels: Guarda los modelos en archivos .h5.
Model:

Clase que define y entrena las redes neuronales usando Keras.
Métodos:
train: Entrena una red neuronal con los datos proporcionados.
predict: Realiza predicciones con datos de entrada.
Archivo DatosFinalFinal.xlsx:

Contiene las mediciones de tiempo y temperatura del sensor.
Visualización:
Los resultados del modelo se grafican para comparar los datos reales con las predicciones.
Utiliza matplotlib y plotly para crear gráficos interactivos.
Organización de módulos:
plaintext
Copiar
Editar
.
├── NNDigitalTwinTemperature.py  # Script principal con el código del proyecto.
├── DatosFinalFinal.xlsx         # Archivo de datos de entrada.
├── modelos                      # Carpeta para guardar los modelos entrenados (.h5).
Uso
Entrenar el modelo: Ejecute el script principal para entrenar el gemelo digital con los datos de entrada.

Guardar los modelos: Los modelos se guardarán automáticamente como archivos .h5 en la carpeta modelos.

Simulaciones: Ejecute simulaciones para predicciones directas e inversas utilizando las funciones disponibles en la clase XNNSensor.