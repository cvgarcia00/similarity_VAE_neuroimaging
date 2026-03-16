# Demographic Statistics Extraction

Este directorio contiene scripts para extraer estadísticas demográficas de los conjuntos de datos de entrenamiento, validación y test para su uso en publicaciones científicas.

## Archivos Generados

### Scripts Python
- **`demographic_stats.py`**: Script básico que extrae estadísticas demográficas simples (edad y sexo)
- **`demographic_stats_extended.py`**: Script extendido que incluye también información de diagnóstico (CN, MCI, AD), educación, etc.

### Archivos de Salida
- **`demographic_statistics.csv`**: Estadísticas básicas en formato CSV
- **`demographic_statistics_extended.csv`**: Estadísticas completas en formato CSV  
- **`demographic_summary.txt`**: Resumen legible con tablas formateadas

## Uso

### Script Básico
```bash
cd /home/crisvgarcia
python BETA_VAE_MODEL/demographic_stats.py
```

Genera:
- Tabla con # sujetos, # scans, edad (media ± std), sexo (M/F)
- Formato LaTeX listo para copiar

### Script Extendido (Recomendado)
```bash
cd /home/crisvgarcia
python BETA_VAE_MODEL/demographic_stats_extended.py
```

Genera:
- Todo lo anterior PLUS:
- Distribución de diagnósticos (CN, MCI, AD)
- Estadísticas de educación
- Tablas LaTeX separadas para demographics y diagnosis
- Información más detallada por split

## Estadísticas Calculadas

### Por cada split (Training/Validation/Test):
- **# Subjects**: Número de sujetos únicos
- **# Scans**: Número total de escaneos/observaciones
- **Age**: Media ± desviación estándar, rango [min, max]
- **Sex**: Distribución por sexo (n y porcentaje)
- **Diagnosis**: Distribución de diagnósticos baseline
  - CN: Cognitively Normal
  - MCI: Mild Cognitive Impairment (incluye EMCI y LMCI)
  - AD: Alzheimer's Disease (incluye Dementia)
  - Other: SMC y otros diagnósticos
- **Education**: Años de educación (media ± std)

## Resultados Actuales

**Dataset Overview:**
- Total unique subjects: **1,632**
- Total scans: **3,466**
- Average scans per subject: **2.12**
- Split ratio: **65% / 15% / 20%** (train/val/test)

**Split Distribution:**

| Split      | # Subjects | # Scans | Age (mean ± std) | Male       | Female     |
|------------|------------|---------|------------------|------------|------------|
| Training   | 1060       | 2239    | 72.8 ± 7.3       | 593 (55.9%)| 467 (44.1%)|
| Validation | 244        | 572     | 73.8 ± 7.0       | 138 (56.6%)| 106 (43.4%)|
| Test       | 328        | 655     | 73.8 ± 7.2       | 175 (53.4%)| 153 (46.6%)|

**Diagnosis Distribution:**

| Split      | CN          | MCI         | AD          |
|------------|-------------|-------------|-------------|
| Training   | 204 (19.2%) | 597 (56.3%) | 197 (18.6%) |
| Validation | 58 (23.8%)  | 122 (50.0%) | 46 (18.9%)  |
| Test       | 78 (23.8%)  | 178 (54.3%) | 47 (14.3%)  |

## Notas Importantes

1. **Sin Data Leakage**: Los sujetos aparecen solo en un split (train, val o test), lo que previene el data leakage incluso cuando hay múltiples observaciones del mismo sujeto.

2. **Splits Balanceados**: Las distribuciones de edad, sexo y educación son similares entre los tres splits, lo que indica un buen proceso de división.

3. **Diagnósticos**: MCI es el diagnóstico predominante (~50-56% de los sujetos), seguido de CN (~19-24%) y AD (~14-19%).

4. **Formato LaTeX**: Ambos scripts generan código LaTeX listo para copiar directamente en tu paper.

## Modificaciones

Si necesitas modificar los splits o agregar más estadísticas:

1. Los splits se definen en `config.yaml` bajo `loader.splits`
2. La información demográfica viene de `ADNI_BIDS/participants.tsv`
3. Puedes editar las funciones `calculate_stats()` en los scripts para añadir más métricas

## Dependencias

- pandas
- numpy
- PyYAML
- Los módulos del proyecto (config.py)

Todos los imports relativos se manejan automáticamente en los scripts.
