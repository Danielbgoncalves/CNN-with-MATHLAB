# Projeto de Transfer Learning com ResNet50 no MATLAB

Este projeto utiliza a ResNet50 pré-treinada para extrair características de imagens das categorias aviões, faces e tartarugas (airplanes, faces e hawksbill) e treina um classificador ECOC para seu reconhecimento. O objetivo é demonstrar como realizar transferência de aprendizado utilizando a Deep Learning Toolbox de um modo que seja divertido ver a sua aplicação. 

O script possui comentários que ajudam a interpretar cada função e passo de modo que o entendimento fique mais claro e possa ajudar possíveis aspirantes a entendedores de CNNs, assim como eu!

## Motivação
A transferencia de aprendizado possibilita aproveitarmos modelos pré-treinados em grandes conjuntos de dados para tarefas específicas, o que reduz o tempo de treinamento e melhora a performance.

## Objetivos
- Demonstrar o pré-processamento e redimensionamento de imagens para redes pré-treinadas.
- Extrair características de uma camada intermediária (fc1000) da ResNet50.
- Treinar um classificador ECOC utilizando os recursos extraídos.

## Como Rodar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seuUsuario/CNN-with-MATHLAB.git

2. Descompacte a pasta caltech-101.zip

3. Abra o MATLAB e navegue ate a pasta do projeto

4. Execute a classifier.m 




## Mão na massa
As coisas são mais legais quando participamos delas então é legal usar uma imagem sua no classificador, testar com uma imagem qualquer para colocarmos a rede neural à prova!

1. Para isso baixe alguma imagem da internet que seja de alguma das classes reconhecidas e adicione no diretório do projeto clonado. Vamos supor que ficou salva como 'teste106.png'
2. Atualize a linha 81 do arquivo classifier.m para corresponder ao caminho da nova imagem, basta mudar o nome.
3. Execute novamente o script e veja-o funcionar.

## Resultados
Após a execução, o classificador atingiu uma acurácia de 100% no conjunto das imagens testadas.

O caltech-101 oferece diversas imagens nas categorias escolhidas, aqui, foram usadas 30% para treinamento e 70% para teste.
## Referências
- [Documentação MATLAB Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/)
- [Sobre o caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)

