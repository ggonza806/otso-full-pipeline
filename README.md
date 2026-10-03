# OTSO Penumbra Mapping Pipeline

Pipeline científico para construir ventanas de rigidez geomagnética a partir de una grilla estimativa de MATLAB, escanear trayectorias con OTSO, verificar el cierre de las ventanas, clasificar los casos y producir métricas y mapas 2D de la estructura interna de la penumbra.

La documentación completa en ruso para colaboradores está en [README_RU.md](README_RU.md).

## Convención física

- `raw_filter == 1`: trayectoria permitida (`white`).
- `raw_filter != 1`: trayectoria prohibida (`black`).

Los casos que continúan abiertos después de agotar sus márgenes de tolerancia se excluyen. No se convierten en ceros físicos.

## Requisitos

- Python 3.12.
- OTSO 1.2.19.
- NumPy, pandas, SciPy y Matplotlib.

Instalación recomendada con Conda:

```bash
conda env create -f environment.yml
conda activate otso_1219
```

Alternativamente:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Datos de entrada

`data/penumbra_boundaries.mat` contiene:

- ejes `latGrid`, `lonGrid`, `zenGrid`, `azGrid`;
- bordes estimativos `RlNet` y `RuNet`;
- orden de ejes `(latitude, longitude, zenith, azimuth)`.

`rigidity_window_grid.py` limpia los bordes e interpola linealmente dentro del dominio original. No extrapola. Los nodos con `NaN`, `Ru = 55 GV` —techo abierto del scan fuente— o inversiones grandes se descartan.

## Configuración del experimento

Antes de iniciar un run, revisar:

- en `rigidity_window_grid.py`: `LAT_OUT`, `LON_OUT`, `ZEN_OUT`, `AZ_OUT`;
- en `5d_raw_scan.py`: fecha, parámetros de OTSO, `INTERPOLATED_H0_VALUES`, paso de rigidez, márgenes y tolerancias;
- `STRATIFIED_TEST_CASES = None` para conservar el plan interpolado completo;
- `MAX_CASES = None` para ejecutar todos los casos del plan.

El MATLAB no contiene una dimensión de altura. La altura se fija explícitamente mediante `INTERPOLATED_H0_VALUES`.

## Ejecución

Ejecutar desde la raíz del repositorio:

```bash
python 5d_raw_scan.py
python 5d_check_closure.py
python 5d_rerun_open_cases.py
python 5d_check_closure_final.py
python 5d_finalize_classification.py
python 5d_compute_penumbra_metrics.py
```

Stage 1 crea el run en `runs/5d_raw_default_requested_scans/` y guarda su ruta en `active_run.txt`. Las etapas siguientes leen automáticamente ese archivo. Ambos están ignorados por Git.

Para procesar un run situado en otra carpeta:

```bash
export PENUMBRA_RUN_DIR=/ruta/absoluta/al/run
```

Para guardar todos los runs fuera del repositorio:

```bash
export PENUMBRA_RUNS_DIR=/ruta/absoluta/a/runs
```

La prioridad es: `PENUMBRA_RUN_DIR` explícito y, si no existe, `active_run.txt`.

## Etapas y productos principales

| Etapa | Script | Acción | Salida principal |
|---|---|---|---|
| Interpolación | `rigidity_window_grid.py` | Limpia e interpola `Rl/Ru` | ventanas densificadas |
| 1 | `5d_raw_scan.py` | Construye el plan y ejecuta OTSO | `case_plan.csv`, `plot_cell_plan.csv`, `raw_barcodes_initial.csv` |
| 2 | `5d_check_closure.py` | Comprueba diez puntos negros abajo y diez blancos arriba | `closure_report_initial.csv`, `open_cases_for_rerun.csv` |
| 3 | `5d_rerun_open_cases.py` | Escanea únicamente extensiones dentro de tolerancia | `raw_barcodes_merged.csv` |
| 3b | `5d_check_closure_final.py` | Reevalúa el cierre y descarta los casos aún abiertos | `closure_report_final.csv` |
| 4 | `5d_finalize_classification.py` | Asigna estado y política métrica | `final_case_classification.csv`, `final_plot_cell_classification.csv` |
| 5 | `5d_compute_penumbra_metrics.py` | Calcula métricas de penumbra | `penumbra_metrics_by_case.csv`, `penumbra_metrics_by_plot_cell.csv` |

Las etapas posteriores a Stage 1 trabajan por `case_id` y no dependen de cómo se produjo la grilla.

## Gráficos

Los heatmaps leen `penumbra_metrics_by_plot_cell.csv`. Métricas admitidas:

`dR_total`, `dR_white`, `dR_black`, `dR_gray`, `f_gray`, `N_100`, `N_400`.

- `plot_geographic_metric_validation.py`: mapa latitud–longitud con altura y dirección fijas.
- `plot_geographic_dR_total_validation.py`: mapa latitud–longitud para una altura elegida; admite cualquier métrica mediante `FIELD_NAME`.
- `plot_angular_metric_validation.py`: mapa azimut–zenit con latitud, longitud y altura fijas.
- `plot_angular_validation.py`: variante de validación angular.
- `export_graficos_vkr.py`: exportación por lotes.
- `plot_barcode_compare.py`: barcode de un caso.

Ejemplo:

```bash
python plot_geographic_metric_validation.py

export PENUMBRA_PLOT_OUTPUT_DIR=/ruta/a/figuras
python export_graficos_vkr.py
```

Sin `PENUMBRA_PLOT_OUTPUT_DIR`, las figuras se guardan en `<run>/plots/`.

Para el mapa geográfico actual se fijan `h0 = 400 km`, `zenith = 0°`, `azimuth = 0°`. Para el mapa angular, las coordenadas geográficas configuradas deben existir exactamente en el plan interpolado. Asunción es únicamente un ejemplo de corte 2D; no es una categoría del pipeline.

## Reanudación y fallos OTSO

OTSO 1.2.19 puede finalizar algún worker nativo de manera abrupta. Cada caso se ejecuta en un subprocess para proteger el proceso principal. Los estados y errores se guardan en CSV. Para continuar Stage 1, establecer `RESUME_RUN_ID` en `5d_raw_scan.py` con el nombre del run incompleto.

## Publicación y reproducibilidad

Los resultados de ejecución, figuras, estados locales y caches están excluidos mediante `.gitignore`; el archivo MATLAB utilizado para construir las ventanas sí está versionado.

La acción de GitHub definida en `.github/workflows/validate.yml` comprueba la sintaxis y reproduce la interpolación del archivo MATLAB en cada `push` y `pull_request`. No ejecuta OTSO, porque el scan completo es un trabajo científico de larga duración.

El código y la documentación de este proyecto se distribuyen bajo la [licencia MIT](LICENSE). OTSO y las demás dependencias conservan sus propias licencias. El responsable del proyecto ha confirmado que se permite redistribuir `data/penumbra_boundaries.mat`; esa autorización no implica atribuirle una licencia MIT ni cambiar los derechos de sus autores.

La publicación puede hacerse con:

```bash
git init -b main
git add .
git commit -m "Initial public release"
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```
