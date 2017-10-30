muestras = 10;
nivel = 10;
linea = 1;
%Para analiss en grados y dstancia
distV = 8;
distH = 8;
%Extraccion de caracteristicas de la imagen D6
for x=1:muestras
    for y=1:muestras
        imgAl = imread('D6.BMP');
        imgAl = rgb2gray(imgAl);
        imgAl = imgAl(1*x:64*x,1*y:64*y);
        imgAl = histeq(imgAl, nivel);
        cocoM = zeros(nivel);
        %Feature extraction
        cocoM = graycomatrix(imgAl, 'offset', [distV,distH]);
        media = mean(mean(imgAl));
        stats = graycoprops(cocoM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F(linea,:) = vectorP;
        L(linea,:) = 'D6.BMP ';
        linea = linea + 1;
    end
end
%Extraccion de caracteristicas de la imagen D64
linea = 1;
for x=1:muestras
    for y=1:muestras
        imgAl = imread('D64.BMP');
        imgAl = rgb2gray(imgAl);
        imgAl = imgAl(1*x:64*x,1*y:64*y);
        imgAl = histeq(imgAl, nivel);
        cocoM = zeros(nivel);
        %Feature extraction
        cocoM = graycomatrix(imgAl, 'offset', [distV,distH]);
        media = mean(mean(imgAl));
        stats = graycoprops(cocoM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F2(linea,:) = vectorP;
        L2(linea,:) = 'D64.BMP';
        linea = linea + 1;
    end
end
%Extraccion de caracteristicas de la imagen 22.tiff
linea = 1;
for x=1:muestras
    for y=1:muestras
        imgAl = imread('22.tiff');
        %imgAl = rgb2gray(imgAl);
        imgAl = imgAl(1*x:32*x,1*y:32*y);
        imgAl = histeq(imgAl, nivel);
        cocoM = zeros(nivel);
        %Feature extraction
        cocoM = graycomatrix(imgAl, 'offset', [distV,distH]);
        media = mean(mean(imgAl));
        stats = graycoprops(cocoM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F3(linea,:) = vectorP;
        L3(linea,:) = '22.tiff';
        linea = linea + 1;
    end
end
%concatenacion de las matrices
Fr = vertcat(F, F2, F3);
Lr = vertcat(L, L2, L3);

NB = NaiveBayes.fit(Fr, Lr);
Mdl = fitcdiscr(Fr, Lr);
KNN = ClassificationKNN.fit(Fr, Lr);


%Clasificador

imgTest = imread('D6.bmp');
imgTest = rgb2gray(imgTest);
imgTest = imgTest(576:640,576:640);
imgTest = histeq(imgTest,nivel);
cocoM = zeros(nivel);
%Feature extraction
cocoM = graycomatrix(imgTest, 'offset', [distV,distH]);
media = mean(mean(imgTest));
stats = graycoprops(cocoM);
%Vector de caracteristicas
vectorF = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
resultadoKNN = predict(KNN,vectorF);
resultadoMdl = predict(Mdl,vectorF);
resultadoNB = predict(NB,vectorF);