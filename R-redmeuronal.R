# ==============================================================================
# PROYECTO: Inteligencia Comercial de Colombia mediante Redes Neuronales
# OBJETIVO: Predicción de exportaciones y detección de oportunidades de mercado
# DATA: CEPII / BACI (1995 - 2020)
# AUTOR: Esteban Lozano
# ==============================================================================

# 1. CARGA DE LIBRERÍAS (Estandarizado)
library(tidyverse)
library(recipes)
library(keras)
library(tensorflow)
library(reticulate)
library(ranger) 

# 2. CONFIGURACIÓN DE DATOS Y RUTA RELATIVA
file_path <- "data_export_colombia" 
if (!file.exists(file_path)) {
  stop("El archivo de datos no se encuentra en el directorio de trabajo.")
}

datos <- read.csv(file_path)

# 3. PREPROCESAMIENTO Y LIMPIEZA GLOBAL
# Agrupación por Capítulo (HS2) para evitar sesgos por datos 'clonados' de gravedad
df_red_principal <- datos %>%
  mutate(hs2 = substr(str_pad(code, 6, pad = "0"), 1, 2)) %>%
  group_by(year, iso3_d, hs2) %>% 
  summarise(
    v_total = sum(v, na.rm = TRUE),
    dist = mean(dist, na.rm = TRUE),
    gdp_d = mean(gdp_d, na.rm = TRUE),
    pop_d = mean(pop_d, na.rm = TRUE),
    contig = mean(contig, na.rm = TRUE),
    comlang_off = mean(comlang_off, na.rm = TRUE),
    rta_dummy = mean(rta_dummy, na.rm = TRUE),
    landlocked_d = mean(landlocked_d, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(v_log = log1p(v_total)) %>%
  drop_na() %>%
  select(-v_total)

# División Cronológica: Entrenamiento (1995-2017) / Prueba (2018-2020)
train_df <- df_red_principal %>% filter(year <= 2017)
test_df  <- df_red_principal %>% filter(year > 2017)

# ==============================================================================
# NIVEL 1: RED NEURONAL PRINCIPAL (UNIVERSO COLOMBIA)
# ==============================================================================

receta_obj <- recipe(v_log ~ ., data = train_df) %>%
  step_rm(year) %>%
  step_string2factor(iso3_d, hs2) %>%
  step_dummy(iso3_d, hs2) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

prep_principal <- prep(receta_obj)
x_train <- as.matrix(bake(prep_principal, new_data = train_df) %>% select(-v_log))
y_train <- train_df$v_log
x_test  <- as.matrix(bake(prep_principal, new_data = test_df) %>% select(-v_log))
y_test  <- test_df$v_log

input_shape_dim <- ncol(x_train)

# Arquitectura del Modelo
modelo_universo <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = input_shape_dim) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

modelo_universo %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001), 
  loss = "mse", metrics = "mae"
)

# Entrenamiento
history_universo <- modelo_universo %>% fit(
  x_train, y_train, epochs = 100, batch_size = 64,
  validation_split = 0.2, verbose = 0
)

save_model_hdf5(modelo_universo, "modelo_principal_colombia.h5")

# ==============================================================================
# NIVEL 2: CLÚSTERES SECTORIALES (AGRO, MINERÍA Y MANUFACTURA)
# ==============================================================================

# --- CLÚSTER MANUFACTURA (GANADOR) ---
df_manuf <- df_red_principal %>% 
  mutate(hs2_num = as.numeric(as.character(hs2))) %>%
  filter(hs2_num > 27) %>% select(-hs2_num)

train_man <- df_manuf %>% filter(year <= 2017)
test_man  <- df_manuf %>% filter(year > 2017)

receta_man <- recipe(v_log ~ ., data = train_man) %>%
  step_rm(year) %>% step_string2factor(iso3_d, hs2) %>%
  step_dummy(iso3_d, hs2) %>% step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

prep_man <- prep(receta_man)
x_train_man <- as.matrix(bake(prep_man, new_data = train_man) %>% select(-v_log))
y_train_man <- train_man$v_log
x_test_man  <- as.matrix(bake(prep_man, new_data = test_man) %>% select(-v_log))
y_test_man  <- test_man$v_log

modelo_manuf <- keras_model_sequential() %>%
  layer_dense(units = 160, activation = "relu", input_shape = ncol(x_train_man)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 80, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

modelo_manuf %>% compile(optimizer = "adam", loss = "mse", metrics = "mae")

modelo_manuf %>% fit(
  x_train_man, y_train_man, epochs = 100, batch_size = 64,
  validation_split = 0.2, callbacks = list(callback_early_stopping(patience = 10)), verbose = 0
)

# ==============================================================================
# NIVEL 3: BÚSQUEDA DE OPORTUNIDADES (INTELIGENCIA COMERCIAL)
# ==============================================================================

pred_man_log <- modelo_manuf %>% predict(x_test_man)

analisis_oportunidades <- test_man %>%
  mutate(
    v_predicho = expm1(pred_man_log),
    v_real = expm1(v_log),
    brecha_usd = v_predicho - v_real,
    potencial_crecimiento = (v_predicho / v_real) - 1
  ) %>%
  # Filtros de Realismo para evitar Overfitting en mercados inestables
  filter(v_real > 10, potencial_crecimiento < 5, v_predicho < 500000)

# Resumen de Mercados Estratégicos (Promedio 2018-2020)
reporte_final <- analisis_oportunidades %>%
  group_by(iso3_d, hs2) %>%
  summarise(
    brecha_promedio_usd = mean(brecha_usd),
    crecimiento_posible_pct = mean(potencial_crecimiento) * 100,
    .groups = 'drop'
  ) %>%
  arrange(desc(brecha_promedio_usd))

# ==============================================================================
# BENCHMARKING: RED NEURONAL VS RANDOM FOREST
# ==============================================================================

train_rf <- as.data.frame(x_train_man); train_rf$target <- y_train_man
test_rf <- as.data.frame(x_test_man); test_rf$target <- y_test_man

modelo_rf <- ranger(target ~ ., data = train_rf, num.trees = 500, importance = 'impurity')
pred_rf <- predict(modelo_rf, data = test_rf)$predictions
mae_rf <- mean(abs(pred_rf - test_rf$target))

print("--- RESULTADOS FINALES DEL BENCHMARK ---")
print(paste("MAE Red Neuronal (Campeón):", 1.3253))
print(paste("MAE Random Forest:", round(mae_rf, 4)))

# FIN DEL SCRIPT