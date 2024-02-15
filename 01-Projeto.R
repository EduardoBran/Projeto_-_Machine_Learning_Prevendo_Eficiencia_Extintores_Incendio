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
library(tidyr)          # manipulação de dados
library(ROSE)           # balanceamento

library(ggplot2)        # gera gráficos
library(patchwork)      # unir gráficos
library(ROCR)           # Gerando uma curva ROC em R

library(randomForest)   # carrega algoritimo de ML (randomForest)
library(e1071)          # carrega algoritimo de ML (SVM)
library(gbm)            # carrega algoritimo de ML (GBM)
library(xgboost)        # carrega algoritimo de ML (XgBoost)
library(neuralnet)      # carrega algoritimo de ML (Redes Neurais)

library(caret)          # cria confusion matrix
library(corrplot)       # análise de correlação

library(h2o)            # framework para construir modelos de machine learning



#  -> Seu trabalho é construir um modelo de Machine Learning capaz de prever, com base em novos dados, se a chama será extinta ou não ao usar um 
#     extintor de incêndio.



#### Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))


#### Análise Exploratória dos Dados

# Verificando e removendo valores ausentes
dados <- dados[complete.cases(dados), ]
colSums(is.na(dados))

# Tipo de Dados
dim(dados)
str(dados)
summary(dados)

length(unique(dados$SIZE))
length(unique(dados$FUEL))
length(unique(dados$DISTANCE))
length(unique(dados$AIRFLOW))
length(unique(dados$FREQUENCY))
length(unique(dados$STATUS))

# Modificando Variáveis Para Tipo Factor
dados$STATUS <- as.factor(dados$STATUS)
dados$SIZE <- as.factor(dados$SIZE)
dados$FUEL <- as.factor(dados$FUEL)
str(dados)
summary(dados)


## Visualizando por Gráficos

# Histogramas para variáveis numéricas
dados_long <- pivot_longer(dados, cols = c("DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY"))

ggplot(dados_long, aes(x = value)) + 
  geom_histogram(bins = 30, fill = "skyblue", color = "black") + 
  facet_wrap(~name, scales = "free", ncol = 2) + 
  xlab("") + ylab("Frequência") + 
  ggtitle("Histogramas de Variáveis Numéricas") + 
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) # Centraliza o título
rm(dados_long)


# Criando o BoxPlot
dados_long <- pivot_longer(dados, cols = c("DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY"), 
                           names_to = "Variable", values_to = "Value")
ggplot(dados_long, aes(x = STATUS, y = Value, fill = STATUS)) + 
  geom_boxplot() + 
  facet_wrap(~Variable, scales = "free_y") + 
  labs(title = "Boxplots por STATUS", y = "Valor", x = "STATUS") +
  scale_fill_manual(values = c("#FF9999", "#9999FF")) + # Cores para os boxplots
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))


# Gráficos de Violino para Variáveis Numéricas por STATUS
ggplot(dados_long, aes(x = STATUS, y = Value, fill = STATUS)) + 
  geom_violin(trim = FALSE) + 
  facet_wrap(~Variable, scales = "free_y") + 
  labs(title = "Gráficos de Violino das Variáveis Numéricas por STATUS", x = "STATUS", y = "Valor") +
  scale_fill_manual(values = c("0" = "#FF9999", "1" = "#9999FF")) +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))


# Primeiro gráfico: Distribuição de SIZE por STATUS
plot_size <- ggplot(dados, aes(x = SIZE, fill = STATUS)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Distribuição de SIZE por STATUS", x = "SIZE", y = "Contagem") +
  scale_fill_manual(values = c("0" = "#FF9999", "1" = "#9999FF")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Segundo gráfico: Distribuição de FUEL por STATUS
plot_fuel <- ggplot(dados, aes(x = FUEL, fill = STATUS)) + 
  geom_bar(position = "dodge") + 
  labs(title = "Distribuição de FUEL por STATUS", x = "FUEL", y = "Contagem") +
  scale_fill_manual(values = c("0" = "#FF9999", "1" = "#9999FF")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Combinando os dois gráficos em um único plot
plot_size / plot_fuel
rm(plot_size)
rm(plot_fuel)






#### Criando Lista Para Armazenar Resultados dos Modelos das Versões
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

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
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
# - Cria 1 Tipo de Modelo utilizando Seleção de Variáveis (RandomForest)


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

rm(previsoes, conf_mat)





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
#resultados_modelos # Acc 0.9584

rm(previsoes, conf_mat)




#### Versão 5

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Aplica Normalização Nas Variáveis Numéricas
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando Seleção de Variáveis (RandomForest)


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



## Criando Modelos

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


# SVM
modelo_svm <- svm(STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, 
                  data = dados_treino, 
                  type = "C-classification", 
                  kernel = "radial")


# GLM
modelo_glm <- glm(data = dados_treino, STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, family = binomial(link = 'logit'))


# Xgboost
dados_treino_xgb <- xgb.DMatrix(data.matrix(dados_treino[,-which(names(dados_treino) == "STATUS")]), label = as.numeric(dados_treino$STATUS)-1)
dados_teste_xgb <- xgb.DMatrix(data.matrix(dados_teste[,-which(names(dados_teste) == "STATUS")]), label = as.numeric(dados_teste$STATUS)-1)

param <- list(
  objective = "binary:logistic", # Objetivo para classificação binária
  booster = "gbtree",            # Uso de árvores de decisão como boosters
  eta = 0.3,                     # Taxa de aprendizado
  max_depth = 6                  # Profundidade máxima de cada árvore
)

modelo_xgb <- xgb.train(
  params = param,
  data = dados_treino_xgb, 
  nrounds = 100
)
rm(param)



## Avaliando e Visualizando Desempenho dos Modelos
previsoes_rf <- predict(modelo_rf, newdata = dados_teste)
previsoes_svm <- predict(modelo_svm, newdata = dados_teste)
previsoes_glm <- predict(modelo_glm, newdata = dados_teste, type = 'response')
previsoes_xgb <- predict(modelo_xgb, newdata = dados_teste_xgb)
conf_mat_rf <- confusionMatrix(previsoes_rf, dados_teste$STATUS)
conf_mat_svm <- confusionMatrix(previsoes_svm, dados_teste$STATUS)
conf_mat_glm <- confusionMatrix(factor(ifelse(previsoes_glm > 0.5, 1, 0)), dados_teste$STATUS)
conf_mat_xgb <- confusionMatrix(factor(ifelse(previsoes_xgb > 0.5, 1, 0)), factor(as.numeric(dados_teste$STATUS) - 1))

resultados_modelos[['Versao5_rf']] <- list(
  Accuracy = round(conf_mat_rf$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat_rf$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat_rf$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat_rf$byClass['Balanced Accuracy'], 4)
)
resultados_modelos[['Versao5_svm']] <- list(
  Accuracy = round(conf_mat_svm$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat_svm$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat_svm$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat_svm$byClass['Balanced Accuracy'], 4)
)
resultados_modelos[['Versao5_glm']] <- list(
  Accuracy = round(conf_mat_glm$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat_glm$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat_glm$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat_glm$byClass['Balanced Accuracy'], 4)
)
resultados_modelos[['Versao5_xgb']] <- list(
  Accuracy = round(conf_mat_xgb$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat_xgb$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat_xgb$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat_xgb$byClass['Balanced Accuracy'], 4)
)


rm(previsoes_rf, previsoes_svm, previsoes_glm, previsoes_xgb)
rm(conf_mat_rf, conf_mat_svm, conf_mat_glm, conf_mat_xgb)
rm(dados_treino_xgb, dados_teste_xgb)




#### Versão 6

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Adicionando Novas Variáveis de Relação
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando Seleção de Variáveis (RandomForest)


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
                       data = dados_nor, 
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
modelo_rf <- randomForest(STATUS ~ Dist_Airflow_Ratio + Freq_Airflow_Inverse + AIRFLOW + DISTANCE
                          + SIZE + Desibel_Freq_Product + FREQUENCY + FUEL, 
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
#resultados_modelos # Acc 0.9578

rm(previsoes, conf_mat)




#### Versão 8

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Converte As Variáveis SIZE, FUEL e a variável alvo STATUS para tipo factor
# - Aplica Normalização Nas Variáveis Numéricas
# - Aplica técnica de Balanceamento da Variável Alvo
# - Aplica Seleção de Variáveis
# - Cria 1 Tipo de Modelo utilizando Seleção de Variáveis (RandomForest)


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


# Balanceamento da Variável Alvo

# Aplicando o SMOTE para balancear a variável alvo
table(dados_nor$STATUS)
dados_balanceados <- ovun.sample(STATUS ~ ., data = dados_nor, method = "over", N = 2*max(table(dados$STATUS)))$data

# Verificando a distribuição da variável alvo após o balanceamento
table(dados_balanceados$STATUS)



## Criando Modelo

# Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(dados_balanceados$STATUS, p = 0.80, list = FALSE)
dados_treino <- dados_balanceados[indices, ]
dados_teste <- dados_balanceados[-indices, ]
rm(indices)

# RandomForest
modelo_rf <- randomForest(STATUS ~ AIRFLOW + DISTANCE + FREQUENCY + SIZE + FUEL, 
                          data = dados_treino, 
                          ntree = 100, nodesize = 10, importance = TRUE)


## Avaliando e Visualizando Desempenho do Modelo
previsoes <- predict(modelo_rf, newdata = dados_teste)
conf_mat <- confusionMatrix(previsoes, dados_teste$STATUS)
conf_mat
resultados_modelos[['Versao8']] <- list(
  Accuracy = round(conf_mat$overall['Accuracy'], 4),
  Sensitivity = round(conf_mat$byClass['Sensitivity'], 4),
  Specificity = round(conf_mat$byClass['Specificity'], 4),
  Balanced_Accuracy = round(conf_mat$byClass['Balanced Accuracy'], 4)
)
# resultados_modelos # Acc 0.9717

rm(previsoes, conf_mat)





#### Versão 9 (AutoMl)

## Carregando dados
dados <- data.frame(read_xlsx("dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"))
dados <- dados[complete.cases(dados), ]
str(dados)

# - Utiliza Configurações da Versão 2 (apenas modificar as variáveis chr para tipo factor)
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
                            max_runtime_secs = 60 * 10,
                            sort_metric = "AUC")                               # Mudança da métrica de avaliação para AUC, adequada para classificação


# Extrai o leaderboard (dataframe com os modelos criados)
leaderboard_automl <- as.data.frame(modelo_automl@leaderboard)
head(leaderboard_automl, 3)
View(leaderboard_automl)


# Extrai os líderes (modelo com melhor performance)
lider_automl_gbm <- modelo_automl@leader
print(lider_automl_gbm)
lider_automl_sta <- h2o.getModel(leaderboard_automl$model_id[2])
print(lider_automl_sta)
lider_automl_xgb <- h2o.getModel(leaderboard_automl$model_id[21])
print(lider_automl_xgb)

# Salvando os Modelos
# h2o.saveModel(lider_automl_gbm, path = "modelos", force = TRUE)
# h2o.saveModel(lider_automl_sta, path = "modelos", force = TRUE)
# h2o.saveModel(lider_automl_xgb, path = "modelos", force = TRUE)

# Carregando os Modelos
modelo_gbm <- h2o.loadModel(path = "modelos/GBM_grid_1_AutoML_1_20240215_131817_model_19")
modelo_sta <- h2o.loadModel(path = "modelos/StackedEnsemble_BestOfFamily_4_AutoML_1_20240215_131817")
modelo_xgb <- h2o.loadModel(path = "modelos/XGBoost_2_AutoML_1_20240215_131817")


# Avaliação dos Modelos
h2o.performance(modelo_gbm, newdata = h2o_frame_split[[2]])  # AUC:  0.9987656
h2o.performance(modelo_sta, newdata = h2o_frame_split[[2]])  # AUC:  0.9987226
h2o.performance(modelo_xgb, newdata = h2o_frame_split[[2]])  # AUC:  0.9973968


h2o.shutdown()




modelo_gbm





# NORMALIZAR VALORES NUMÉRICOS (feito)
# CRIAR NOVAS VARIAVEIS DE RELAÇÃO (feito)

## TRANSFORMAR VARIÁVEIS NUMERICAS EM CATEGORICAS ATRAVELS DE LEVELS (feito)

### ->> BALANCEAR A VARIÁVEL ALVO USANDO A VERSÃO 8 (feito)


#### ADICIONAR OS NOVOS ALGORITMOS DA VERSÃO 5 NA VERSÃO 8 


##### Após escolha da melhor versão de dados -> SELEÇÃO DE VARIÁVEIS (criar um loop para testar todas as combinações)












## Adicionando Resultados das Versões em um DataFrame
modelos_params <- do.call(rbind, lapply(resultados_modelos, function(x) data.frame(t(unlist(x))))) # Convertendo a lista de resultados em um dataframe
modelos_params
modelos_params <- 
  data.frame(Version = names(resultados_modelos), do.call(rbind, lapply(resultados_modelos, function(x) unlist(x))), row.names = NULL)
View(modelos_params)








# 1. Codificação One-Hot para Variáveis Categóricas FUEL
# Embora SIZE e STATUS já estejam convertidos para fatores, FUEL sendo categórico pode se beneficiar da codificação one-hot, criando variáveis
# binárias para cada tipo de combustível. Isso é particularmente útil para modelos que não lidam bem com variáveis categóricas.

# CRIAR OUTROS MODELOS COMO XGB



