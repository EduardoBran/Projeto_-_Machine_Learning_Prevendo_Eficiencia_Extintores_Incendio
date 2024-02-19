####  Projeto - Prevendo o Consumo de Energia de Carros Elétricos  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/20.Projetos_com_Feedback/2.Prevendo_Eficiencia_Extintores_Incendio")
getwd()



########################    Machine Learning em Prevenção da Eficiência de Extintores de Incêndio    ########################


## Sobre o Script

# - Este script contém o Storytelling a respeito do projeto + Interface Gráfica



## Carregando Pacotes
library(readxl)         # carregar arquivos

library(tibble)         # manipulação de arquivos

library(ggplot2)        # gera gráficos
library(shiny)          # intercace gráfica

library(xgboost)        # carrega algoritimo de ML (XgBoost)



#################################        STORYTELLING        #################################


# Desafio Inovador na Segurança de Extintores

# - O projeto enfrentou um desafio inovador e altamente relevante na área de segurança e eficiência de extintores de incêndio, ao utilizar dados 
#   experimentais para o desenvolvimento de modelos preditivos. Esses modelos são capazes de prever a eficácia na extinção de chamas através de ondas
#   sonoras. 
# - Iniciando com uma análise exploratória detalhada, que abrangeu desde a limpeza dos dados até a conversão de variáveis categóricas em fatores,
#   o projeto avançou por oito versões distintas de modelagem e preparação de dados. Essas versões variaram desde o tratamento direto dos dados até a
#   aplicação de técnicas avançadas, como normalização, balanceamento de classes e seleção de variáveis, culminando na aplicação de múltiplos modelos de
#   machine learning, como RandomForest, SVM, GLM e XGBoost.

# Evolução e Sofisticação do Modelo

# - Com uma evolução marcada pela complexidade e sofisticação, o projeto demonstrou um compromisso contínuo com a melhoria da precisão e da capacidade
#   de generalização dos modelos. Notavelmente, a introdução de técnicas de balanceamento de dados na Versão 8 ilustrou uma compreensão profunda dos 
#   desafios inerentes ao treinamento de modelos preditivos em conjuntos de dados desequilibrados. O uso do XGBoost, ajustado com hiperparâmetros 
#   otimizados e acompanhado de uma seleção minuciosa de variáveis, destacou-se ao apresentar as métricas de desempenho mais elevadas entre todas as 
#   técnicas examinadas.

# Justificativa da Escolha do Modelo e Interface Gráfica

# - O processo iterativo e fundamentado de modelagem é revelado através do storytelling deste projeto. A escolha do modelo XGBoost na Versão 8, que
#   alcançou uma acurácia impressionante de 0.9832, foi justificada não apenas pelas suas métricas excepcionais de precisão, sensibilidade,
#   especificidade e acurácia balanceada, mas também pela sua habilidade em processar eficientemente a complexidade dos dados e identificar padrões
#   significativos essenciais para previsões confiáveis. Esta escolha reflete um equilíbrio ideal entre complexidade técnica e praticidade, assegurando
#   que o modelo atue como uma ferramenta valiosa na prevenção de incêndios e no aprimoramento da segurança dos extintores.

# - Além disso, para facilitar o acesso e a usabilidade deste modelo avançado, foi desenvolvida uma interface gráfica para usuários.
#   Esta interface permite que indivíduos, sem conhecimento técnico em machine learning, testem e verifiquem a eficácia do modelo em prever a extinção
#   de chamas, tornando o projeto não apenas uma conquista técnica mas também uma solução prática e acessível para a indústria.

#Conclusão e Potencial de Futuras Inovações

# - Em síntese, este projeto serve como um exemplo exemplar de como o machine learning pode ser aplicado com sucesso para solucionar problemas complexos
#   em contextos industriais, ressaltando a importância de abordagens metodológicas e adaptáveis no desenvolvimento de soluções preditivas.
# - O êxito do projeto sublinha o potencial significativo do machine learning em contribuir para a segurança e eficiência dos sistemas de combate a 
#   incêndios, pavimentando o caminho para inovações futuras no campo.




#################################        INTERFACE GRÁFICA        #################################


# Definir a UI
ui <- fluidPage(
  titlePanel("Consulta de Eficiência do Extintor"),
  sidebarLayout(
    sidebarPanel(
      selectInput("size", "Tamanho do Extintor", choices = c("1", "2", "3", "4", "5", "6", "7")),
      selectInput("fuel", "Tipo de Combustível", choices = c("gasoline", "kerosene", "lpg", "thinner")),
      numericInput("distance", "Distância (cm)", value = 100),
      numericInput("frequency", "Frequência", value = 50),
      actionButton("consultar", "Consultar")
    ),
    mainPanel(
      textOutput("resultado")
    )
  )
)

# Definir o servidor
server <- function(input, output) {
  observeEvent(input$consultar, {
    # Criar um novo data frame com os valores de entrada
    novo_dado <- data.frame(SIZE = as.factor(input$size),
                            FUEL = as.factor(input$fuel),
                            DISTANCE = as.numeric(input$distance),
                            FREQUENCY = as.numeric(input$frequency))
    
    # Normalizar a coluna DISTANCE
    novo_dado$DISTANCE <- (novo_dado$DISTANCE - 10) / (190 - 10)
    # Normalizar a coluna FREQUENCY
    novo_dado$FREQUENCY <- (novo_dado$FREQUENCY - 1) / (75 - 1)
    
    # Definir os níveis para as variáveis categóricas
    levels(novo_dado$SIZE) <- c("1", "2", "3", "4", "5", "6", "7")
    levels(novo_dado$FUEL) <- c("gasoline", "kerosene", "lpg", "thinner")
    
    # Criar um conjunto de dados de referência
    ref_data <- expand.grid(SIZE = levels(novo_dado$SIZE),
                            FUEL = levels(novo_dado$FUEL),
                            DISTANCE = 0,  # Valor placeholder
                            FREQUENCY = 0)  # Valor placeholder
    
    # Adicionar a linha de entrada ao conjunto de dados de referência
    ref_data <- rbind(ref_data, novo_dado)
    
    # Converter variáveis categóricas para dummy usando o conjunto de dados de referência
    ref_data_matriz <- model.matrix(~ SIZE + FUEL + DISTANCE + FREQUENCY - 1, data = ref_data)
    
    # Usar apenas a última linha (os dados de entrada) para a previsão
    entrada_xgb <- xgb.DMatrix(data = ref_data_matriz[nrow(ref_data_matriz), , drop = FALSE])
    
    # Carregar o modelo previamente treinado a cada vez que o botão é pressionado
    modelo_carregado <- xgb.load("modelos/modelo_xgb3.model")
    
    # Fazer a previsão
    previsao <- predict(modelo_carregado, entrada_xgb)
    resultado <- ifelse(previsao > 0.5, "SIM", "NÃO")
    print(resultado)
    # Exibir o resultado
    output$resultado <- renderText({
      paste("O extintor apagará o fogo?", resultado)
    })
  })
}

# Executar a aplicação
shinyApp(ui = ui, server = server)


