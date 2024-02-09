####  Projeto - Prevendo o Consumo de Energia de Carros Elétricos  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/20.Projetos_com_Feedback/2.Prevendo_Eficiencia_Extintores_Incendio")
getwd()



########################    Machine Learning em Prevenção da Eficiência de Extintores de Incêndio    ########################


## Sobre o Script

# - Este script contém o código com todas as versões de preparação e criação de modelos para conclusão do projeto



## Carregando Pacotes
library(readxl)         # carregar arquivos

library(dplyr)          # manipulação de dados

library(ggplot2)        # gera gráficos
library(ROCR)           # Gerando uma curva ROC em R

library(randomForest)   # carrega algoritimo de ML (randomForest)
library(e1071)          # carrega algoritimo de ML (SVM)

library(caret)          # cria confusion matrix
library(corrplot)       # análise de correlação

library(h2o)            # framework para construir modelos de machine learning



#  -> Seu trabalho é construir um modelo de Machine Learning capaz de prever, com base em novos dados, se a chama será extinta ou não ao usar um 
#     extintor de incêndio.



#### Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))

# Verificando e removendo valores ausentes
dados <- dados[complete.cases(dados), ]
colSums(is.na(dados))

dim(dados)
str(dados)
summary(dados)

length(unique(dados$SIZE))
length(unique(dados$FUEL))
length(unique(dados$DISTANCE))
length(unique(dados$AIRFLOW))
length(unique(dados$FREQUENCY))
length(unique(dados$STATUS))





## Criando Lista Para Armazenar Resultados dos Modelos das Versões
resultados_modelos <- list()


#### Versão 1

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]


# - Converte apenas a variável alvo para tipo factor
# - Mantem outros tipos de dados originais (chr e num)
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo variável alvo
dados$STATUS <- as.factor(dados$STATUS)
str(dados)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ ., 
                       data = dados_treino, 
                       ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)

resultados_modelos[['Versao1']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos  # Acc 0.9541

rm(previsoes)
rm(conf_mat)





#### Versão 2

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo facotr
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ ., 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)

resultados_modelos[['Versao2']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos  # Acc 0.9613

rm(previsoes)
rm(conf_mat)





#### Versão 3

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(STATUS ~ ., 
                       data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)
names(dados_treino)

# RandomForest
modelo_rf <- randomForest(STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao3']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos # Acc 0.9667

rm(previsoes)
rm(conf_mat)





#### Versão 4

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Cria Novas Variáveis Categórias a partir das Variáves Numéricas
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)

# Criando Novas Variáveis Categóricas com base nos quartis
dados$DISTANCE_f <- cut(dados$DISTANCE, breaks=quantile(dados$DISTANCE, probs=0:4/4, na.rm=TRUE), include.lowest=TRUE, labels=c("D1", "D2", "D3", "D4"))
dados$DESIBEL_f <- cut(dados$DESIBEL, breaks=quantile(dados$DESIBEL, probs=0:4/4, na.rm=TRUE), include.lowest=TRUE, labels=c("DB1", "DB2", "DB3", "DB4"))
dados$AIRFLOW_f <- cut(dados$AIRFLOW, breaks=quantile(dados$AIRFLOW, probs=0:4/4, na.rm=TRUE), include.lowest=TRUE, labels=c("AF1", "AF2", "AF3", "AF4"))
dados$FREQUENCY_f <- cut(dados$FREQUENCY, breaks=quantile(dados$FREQUENCY, probs=0:4/4, na.rm=TRUE), include.lowest=TRUE, labels=c("F1", "F2", "F3", "F4"))


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(STATUS ~ ., 
                       data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ ., 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao4']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos # Acc 0.9332

rm(previsoes)
rm(conf_mat)




#### Versão 5

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Aplica Normalização Nas Variáveis Numéricas
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)

# Aplicando Normalização nas Variáveis Numéricas
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))



## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados_nor$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao5']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos # Acc 0.9624

rm(previsoes)
rm(conf_mat)




#### Versão 6

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Adicionando Novas Variáveis de Relação
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)

# Adicionando Novas Variáveis de Relação (.Machine$double.eps = pequeno valor (conhecido como epsilon) ao denominador para garantir que nunca seja zero.)

# 1. Relação Entre DISTANCE e AIRFLOW
dados$Dist_Airflow_Ratio <- dados$DISTANCE / (dados$AIRFLOW + .Machine$double.eps)

# 2. Produto de DESIBEL e FREQUENCY
dados$Desibel_Freq_Product <- dados$DESIBEL * dados$FREQUENCY

# 3. Relação Inversa Entre FREQUENCY e AIRFLOW
dados$Freq_Airflow_Inverse <- dados$FREQUENCY / (dados$AIRFLOW + .Machine$double.eps)

# 4. Agrupamento de DISTANCE
dados$Distance_Category <- cut(dados$DISTANCE, breaks=c(0, 50, 100, 150, Inf), labels=c("Close", "Moderate", "Far", "Very Far"), include.lowest=TRUE)

# 5. Logaritmo de DESIBEL
dados$Desibel_Log <- log(dados$DESIBEL + .Machine$double.eps)


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(STATUS ~ ., 
                       data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados[indices, ]
dados_teste <- dados[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ AIRFLOW + Dist_Airflow_Ratio + Freq_Airflow_Inverse
                          + DISTANCE + SIZE + FREQUENCY + Desibel_Freq_Product + FUEL + Distance_Category, 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao6']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos # Acc 0.9624

rm(previsoes)
rm(conf_mat)





#### Versão 7

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Adicionando Novas Variáveis de Relação
# - Aplica Normalização Nas Variáveis Numéricas
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando todas as variáveis (RandomForest)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)

# Adicionando Novas Variáveis de Relação (.Machine$double.eps = pequeno valor (conhecido como epsilon) ao denominador para garantir que nunca seja zero.)

# 1. Relação Entre DISTANCE e AIRFLOW
dados$Dist_Airflow_Ratio <- dados$DISTANCE / (dados$AIRFLOW + .Machine$double.eps)

# 2. Produto de DESIBEL e FREQUENCY
dados$Desibel_Freq_Product <- dados$DESIBEL * dados$FREQUENCY

# 3. Relação Inversa Entre FREQUENCY e AIRFLOW
dados$Freq_Airflow_Inverse <- dados$FREQUENCY / (dados$AIRFLOW + .Machine$double.eps)

# 4. Agrupamento de DISTANCE
dados$Distance_Category <- cut(dados$DISTANCE, breaks=c(0, 50, 100, 150, Inf), labels=c("Close", "Moderate", "Far", "Very Far"), include.lowest=TRUE)

# 5. Logaritmo de DESIBEL
dados$Desibel_Log <- log(dados$DESIBEL + .Machine$double.eps)


# Aplicando Normalização nas Variáveis Numéricas
numeric_columns <- sapply(dados, is.numeric)
dados_nor <- dados %>%
  mutate(across(where(is.numeric), ~ scale(., center = min(.), scale = max(.) - min(.))))
rm(numeric_columns)

# Reverter Normalização
# dados_revertidos <- dados_nor %>%
#   mutate(across(where(is.numeric), ~ (. * (max(dados[, cur_column()]) - min(dados[, cur_column()])) + min(dados[, cur_column()]))))


## Seleção de Variáveis (Feature Selection)
modelo <- randomForest(STATUS ~ ., 
                       data = dados, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando por Gráficos
varImpPlot(modelo)

importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 
df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))

rm(modelo)
rm(importancia_ordenada)
rm(df_importancia)


## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados_nor$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados_nor[indices, ]
dados_teste <- dados_nor[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao7']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
#resultados_modelos # Acc 0.9624

rm(previsoes)
rm(conf_mat)




#### Versão 8 (AutoMl)

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Utiliza Configurações das Versões 6 e 7
# - Adiciona AutoMl


## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)


## Preparação dos Dados

# Convertendo as variáveis
dados[c("SIZE", "FUEL", "STATUS")] <- lapply(dados[c("SIZE", "FUEL", "STATUS")], as.factor)
dim(dados)
str(dados)
summary(dados)

# Adicionando Novas Variáveis de Relação (.Machine$double.eps = pequeno valor (conhecido como epsilon) ao denominador para garantir que nunca seja zero.)

# 1. Relação Entre DISTANCE e AIRFLOW
dados$Dist_Airflow_Ratio <- dados$DISTANCE / (dados$AIRFLOW + .Machine$double.eps)

# 2. Produto de DESIBEL e FREQUENCY
dados$Desibel_Freq_Product <- dados$DESIBEL * dados$FREQUENCY

# 3. Relação Inversa Entre FREQUENCY e AIRFLOW
dados$Freq_Airflow_Inverse <- dados$FREQUENCY / (dados$AIRFLOW + .Machine$double.eps)

# 4. Agrupamento de DISTANCE
dados$Distance_Category <- cut(dados$DISTANCE, breaks=c(0, 50, 100, 150, Inf), labels=c("Close", "Moderate", "Far", "Very Far"), include.lowest=TRUE)

# 5. Logaritmo de DESIBEL
dados$Desibel_Log <- log(dados$DESIBEL + .Machine$double.eps)


## Automl

# Inicializando o H2O (Framework de Machine Learning)
h2o.init()

# O H2O requer que os dados estejam no formato de dataframe do H2O
h2o_frame <- as.h2o(dados)
class(h2o_frame)

# Split dos dados em treino e teste (cria duas listas)
h2o_frame_split <- h2o.splitFrame(h2o_frame, ratios = 0.85)
head(h2o_frame_split)
h2o_frame_split


modelo_automl <- h2o.automl(y = 'STATUS',                                      # Nome da variável alvo atualizado para 'STATUS'
                            training_frame = h2o_frame_split[[1]],             # Conjunto de dados de treinamento
                            leaderboard_frame = h2o_frame_split[[2]],          # Conjunto de dados para a leaderboard
                            max_runtime_secs = 60 * 15,
                            sort_metric = "AUC")                               # Mudança da métrica de avaliação para AUC, adequada para classificação

# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
head(leaderboard_automl, 3)
View(leaderboard_automl)


# Extrai o líder (modelo com melhor performance)
lider_automl <- modelo_automl@leader
print(lider_automl)
View(lider_automl)

# Avaliação dos Modelos
h2o.performance(lider_automl, newdata = h2o_frame_split[[2]])

## Calcular Acurácia
# Obter o valor de threshold para F1-optimal a partir do objeto de performance
perf <- h2o.performance(model = lider_automl, newdata = h2o_frame_split[[2]])
perf
optimal_threshold <- h2o.metric(perf)$threshold[which.max(thresholds_and_metrics$f1)]

# Agora, usar o valor de threshold para calcular a acurácia
accuracy <- h2o.accuracy(perf, threshold = optimal_threshold)[1]
accuracy




# Avaliação Modelo AutoMl 1
# accuracy: 0.9832954











# NORMALIZAR VALORES NUMÉRICOS (feito)
# CRIAR NOVAS VARIAVEIS DE RELAÇÃO
# TRANSFORMAR VARIAVSEIS NUMERICAS EM CATEGORICAS ATRAVELS DE LEVELS (feito)
# SELEÇÃO DE VARIÁVEIS (criar um loop para testar todas as combinações)

# CONTINUAR NA VERSÃO 7
# VERIFICAR QUAIS VARIAVEIS NUMÉRICAS SERÃO NORMALIZADAS (PROVAVELMENTE AS ORIGINAIS)

# BALANCEAR A VARIÁVEL ALVO USANDO A VERSÃO 7 OU 8






## Adicionando Resultados das Versões em um DataFrame
modelos_params <- do.call(rbind, lapply(resultados_modelos, function(x) data.frame(t(unlist(x))))) # Convertendo a lista de resultados em um dataframe
modelos_params
modelos_params <- 
  data.frame(Version = names(resultados_modelos), do.call(rbind, lapply(resultados_modelos, function(x) unlist(x))), row.names = NULL)
View(modelos_params)







## ADICIONAR NOVAS VARIÁVEIS DE RELAÇÃO NAS VERSÕES 6, 7 e 8


# 1. Relação Entre DISTANCE e AIRFLOW
dados$Dist_Airflow_Ratio <- dados$DISTANCE / dados$AIRFLOW

# 2. Produto de DESIBEL e FREQUENCY
dados$Desibel_Freq_Product <- dados$DESIBEL * dados$FREQUENCY

# 3. Relação Inversa Entre FREQUENCY e AIRFLOW
dados$Freq_Airflow_Inverse <- dados$FREQUENCY / (dados$AIRFLOW + .Machine$double.eps)

# 4. Agrupamento de DISTANCE
dados$Distance_Category <- cut(dados$DISTANCE, breaks=c(0, 50, 100, 150, Inf), labels=c("Close", "Moderate", "Far", "Very Far"), include.lowest=TRUE)

# 5. Logaritmo de DESIBEL
dados$Desibel_Log <- log(dados$DESIBEL + epsilon)












# 1. Codificação One-Hot para Variáveis Categóricas FUEL
# Embora SIZE e STATUS já estejam convertidos para fatores, FUEL sendo categórico pode se beneficiar da codificação one-hot, criando variáveis
# binárias para cada tipo de combustível. Isso é particularmente útil para modelos que não lidam bem com variáveis categóricas.

# 2. Padronização de Variáveis Numéricas (versao 5)
# Variáveis como DISTANCE, DESIBEL, AIRFLOW, e FREQUENCY devem ser padronizadas para ter uma média de 0 e um desvio padrão de 1.
# Isso é especialmente útil para algoritmos sensíveis à escala das variáveis, como SVM ou kNN.

# 3. Engenharia de Atributos (veraso 4)
# Atributos Polinomiais e de Interação: Considere criar atributos polinomiais para capturar relações não lineares, bem como atributos de 
#                                       interação entre variáveis como DISTANCE * AIRFLOW ou DESIBEL * FREQUENCY.
# Agrupamento por SIZE ou FUEL: Use métodos de agrupamento para identificar padrões dentro de cada categoria de SIZE ou FUEL e criar novas 
#                               variáveis que representem esses grupos.

# 4. Seleção de Atributos (versao 3 e 5)
# Utilize técnicas como importância de variáveis do Random Forest, análise de componentes principais (PCA) para redução de dimensionalidade, 
# ou seleção de variáveis baseada em modelos (como LASSO) para identificar e reter apenas as variáveis mais relevantes.

# 5. Tratamento de Dados Desbalanceados
# A variável alvo STATUS parece estar relativamente balanceada, mas se houver desbalanceamento em novos dados, técnicas como SMOTE para
# oversampling da classe minoritária ou técnicas de undersampling da classe majoritária podem ser consideradas.

# 6. Validação Cruzada Estratificada
# Ao dividir os dados em conjuntos de treino e teste, use a validação cruzada estratificada para garantir que a proporção da variável alvo
# STATUS seja mantida em todos os conjuntos. Isso é importante para modelos de classificação em dados desbalanceados.

# 7. Análise de Correlação
# Realize uma análise de correlação entre as variáveis numéricas e com a variável alvo para identificar possíveis relações lineares fortes que
# possam ser exploradas ou que indiquem redundância entre variáveis.

# 8. Tratamento de Outliers
# Analise a distribuição das variáveis numéricas para identificar e tratar outliers, se necessário, o que pode melhorar a performance do modelo.

# Implementando essas melhorias e estratégias de transformação, você pode aumentar a capacidade do modelo de capturar a complexidade dos dados 
# e, potencialmente, melhorar a precisão na previsão da eficácia dos extintores de incêndio.






