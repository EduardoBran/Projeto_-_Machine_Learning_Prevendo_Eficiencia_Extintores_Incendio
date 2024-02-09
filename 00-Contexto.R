####  PROJETO COM FEEDBACK  (Contexto) ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/20.Projetos_com_Feedback/2.Prevendo_Eficiencia_Extintores_Incendio")
getwd()



########################   Machine Learning em Prevenção da Eficiência de Extintores de Incêndio   ########################


## Sobre o Script

# - Este script contém a história e o contexto do projeto.



## Contexto:

# - O teste hidrostático extintor é um procedimento estabelecido pelas normas da ABNT NBR 12962/2016, que determinam que todos os extintores devem ser
#   testados a cada cinco anos, com a finalidade de identificar eventuais vazamentos, além de também verificar a resistência do material do extintor.

# - Com isso, o teste hidrostático extintor pode ser realizado em baixa e alta pressão, de acordo com estas normas em questão. O procedimento é 
#   realizado por profissionais técnicos da área e com a utilização de aparelhos específicos e apropriados para o teste, visto que eles devem fornecer
#   resultados com exatidão.

# - Seria possível usar Machine Learning para prever o funcionamento de um extintor de incêndio com base em simulações feitas em computador e assim
#   incluir uma camada adicional de segurança nas operações de uma empresa? Esse é o objetivo do Projeto.

# - Usando dados reais disponíveis publicamente, seu trabalho é desenvolver um modelo de Machine Learning capaz de prever a eficiência de extintores
#   de incêndio.


## Dados:

# - No link abaixo você encontra os dados necessários para o seu trabalho:

#  -> https://www.muratkoklu.com/datasets/vtdhnd07.php

# - O conjunto de dados foi obtido como resultado dos testes de extinção de quatro chamas de combustíveis diferentes com um sistema de extinção de
#   ondas sonoras. O sistema de extinção de incêndio por ondas sonoras consiste em 4 subwoofers com uma potência total de 4.000 Watts.
#   Existem dois amplificadores que permitem que o som chegue a esses subwoofers como amplificado. A fonte de alimentação que alimenta o sistema e
#   o circuito do filtro garantindo que as frequências de som sejam transmitidas adequadamente para o sistema está localizada dentro da unidade de
#   controle. Enquanto o computador é usado como fonte de frequência, o anemômetro foi usado para medir o fluxo de ar resultante das ondas sonoras
#   durante a fase de extinção da chama e um decibelímetro para medir a intensidade do som. Um termômetro infravermelho foi utilizado para medir a 
#   temperatura da chama e da lata de combustível, e uma câmera é instalada para detectar o tempo de extinção da chama.

# - Um total de 17.442 testes foram realizados com esta configuração experimental. Os experimentos foram planejados da seguinte forma:

#  -> Três diferentes combustíveis líquidos e combustível GLP foram usados para criar a chama.
#  -> Cinco tamanhos diferentes de latas de combustível líquido foram usados para atingir diferentes tamanhos de chamas.
#  -> O ajuste de meio e cheio de gás foi usado para combustível GLP.

# - Durante a realização de cada experimento, o recipiente de combustível, a 10cm de distância, foi movido para frente até 190cm, aumentando a distância
#   em 10cm a cada vez. Junto com o recipiente de combustível, o anemômetro e o decibelímetro foram movidos para frente nas mesmas dimensões.

# - Experimentos de extinção de incêndio foram conduzidos com 54 ondas sonoras de frequências diferentes em cada distância e tamanho de chama.


## Objetivo:

# - Ao longo dos experimentos de extinção de chama, os dados obtidos de cada dispositivo de medição foram registrados e um conjunto de dados foi criado.
#   O conjunto de dados inclui as características do tamanho do recipiente de combustível representado o tamanho da chama, tipo de combustível,
#   frequência, decibéis, fluxo de ar e extinção da chama. Assim, 6 recursos de entrada e 1 recurso de saída serão usados no modelo que você vai
#   construir.

# - A coluna de status (extinção de chama ou não extinção da chama) pode ser prevista usando os 6 recursos de entrada no conjunto de dados. Os recursos
#   de statuso e combutsível são categóricos, enquanto outros recursos são numéricos.

# - Seu trabalho é construir um modelo de Machine Learning capaz de prever, com base em novos dados, se a chama será extinta ou não ao usar um 
#   extintor de incêndio.


## Explicação das Variáveis:
  
# -> TAMANHO (SIZE): Representa o tamanho da chama por meio do tamanho do recipiente de combustível. Para combustíveis líquidos, os tamanhos
#                    variam de 7 cm a 20 cm e são codificados de 1 a 5. Para GLP, são usadas configurações de meio e completo gás, codificadas
#                    como 6 (meio) e 7 (completo).

# -> COMBUSTÍVEL (FUEL)    : Tipo de combustível utilizado, podendo ser gasolina, querosene, thinner ou GLP.
# -> DISTÂNCIA (DISTANCE)  : Distância entre o sistema de som e o recipiente de combustível, variando de 10 cm a 190 cm.
# -> DECIBÉIS (DESIBEL)    : Intensidade do som produzido pelo sistema de som, medida em decibéis, variando de 72 a 113 dB.
# -> FLUXO de AR (AIRFLOW) : Fluxo de ar resultante das ondas sonoras durante a extinção da chama, medido em metros por segundo (m/s), variando de 0 a 17 m/s.
# -> FREQUÊNCIA (FREQUENCY): Frequência das ondas sonoras usadas para tentar extinguir a chama, variando de 1 a 75 Hz.
# -> STATUS (STATUS)       : Estado da chama após a tentativa de extinção, onde 0 indica não extinção e 1 indica extinção.
