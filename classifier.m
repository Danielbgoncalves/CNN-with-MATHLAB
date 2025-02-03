% Carrega os path usados
mainFolder = fullfile('caltech-101');
rootFolder = fullfile('caltech-101/', '101_ObjectCategories/');

% Vamos treinar com essas imagens
categories = {'airplanes', 'faces', 'hawksbill'};

% Carrega essas imagens
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource','foldernames');

% Garante que usara a mesma quantidade de cada classe de imagem
table = countEachLabel(imds);
minSetCount = min(table{:,2});

imds = splitEachLabel(imds,minSetCount, 'randomized');
countEachLabel(imds);

% Mostrar algumas das imagens na tela
airplanes = find(imds.Labels == 'airplanes', 1);
faces = find(imds.Labels == 'faces', 1);
hawksbills = find(imds.Labels == 'hawksbill', 1);

% figure
% subplot(2,2,1);
% imshow(readimage(imds,airplanes));
% subplot(2,2,2);
% imshow(readimage(imds, faces));
% subplot(2,2,3);
% imshow(readimage(imds, hawksbills));

% Carrega a resnet (modelo pre treinado)
%resnet =  imagePretrainedNetwork('resnet50');
resnet = resnet50();

% Mostra a arquitetura 
% figure
% plot(resnet)
% title('Architecture of resnet 50')
% set(gca, 'YLim', [150 170])

% Separar as imagens em grupos de treino e de teste
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomized');

imageInputSize = resnet.Layers(1).InputSize;

% Pre processa as imagens da caltech para serem compatíveis com o esperado
% pelo modelo
augmentedTrainingSet = augmentedImageDatastore(imageInputSize, ...
    trainingSet, 'ColorPreprocessing','gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageInputSize, ...
    testSet, 'ColorPreprocessing','gray2rgb');

% Camada logo antes da Softmax: boa representação das imagens
featureLayer = 'fc1000';

% Passa as iamgens de 'augmentedTrainingSet' na resnet até a camada
% featureLayer: 
% Processa as imagens em lotes de 32 (mais economico)
% Cada coluna do resultado é um vetor de caracteristicas de uma imagem
trainingFeatures = activations(resnet, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');
trainingLabels = trainingSet.Labels;

% Cria um classificador a partir da resnet 
identifier = fitcecoc(trainingFeatures, trainingLabels, 'Learner',...
    'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% O mesmo mas para o testeSet agora
testFeatures = activations(resnet, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');
testLabels = testSet.Labels;

% Testar a acuracia do classificador (chamdo de identifier aqui)
predictLabels = predict(identifier, testFeatures, 'ObservationsIn', 'columns');
confMat = confusionmat(testLabels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
mean(diag(confMat));

% Testando com imagens que achei na internet
newImage = imread(fullfile("teste105.jpg"));

readImage = augmentedImageDatastore(imageInputSize, ...
    newImage, 'ColorPreprocessing','gray2rgb');

imageFeatures = activations(resnet, ...
    readImage, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

label = predict(identifier, imageFeatures, 'ObservationsIn', 'columns');

sprintf('A imagem carregada é da classe: %s', label)



