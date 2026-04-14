# Documentacion de la comparativa global

## Acrónimos

- **HE**: Histogram Equalization.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization.
- **WTH**: White Top-Hat.
- **BTH**: Black Top-Hat.
- **AMBE**: Absolute Mean Brightness Error. Mide cuánto cambia el brillo medio.
- **PSNR**: Peak Signal-to-Noise Ratio. Mide la similitud entre la imagen original y la procesada.
- **Std. dev.**: Desviación estándar. Se usa como aproximación del contraste global.

## Cómo se calcula la comparativa

Cada algoritmo se evalúa sobre todas las imágenes JPG de la carpeta seleccionada.
Para cada algoritmo se calculan promedios de:

- desviación estándar original
- desviación estándar procesada
- AMBE
- PSNR

También se calcula:

- `std_delta = avg_processed_std - avg_original_std`
- `contrast_effect`:
  - `increase` si `std_delta > 0`
  - `decrease` si `std_delta < 0`
  - `neutral` si `std_delta = 0`

## Ranking

El ranking usa un puntaje compuesto:

`ranking_score = (avg_processed_std * 0.5) + (avg_psnr * 0.05) - (avg_ambe * 0.5)`

Interpretación:

- subir el contraste ayuda
- bajar AMBE ayuda, porque preserva mejor el brillo medio
- subir PSNR ayuda, porque indica menor distorsión respecto a la imagen original

El orden final se obtiene ordenando de mayor a menor `ranking_score`.
