import cv2
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostra = 25
id = input("Identifique - se: ")
largura, altura = 200, 200
print("Capturando as fotos....")

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesdetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize = (100, 100))

    for(x,y,l,a) in facesdetectadas:

        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for(ox,oy,ol,oa) in olhosDetectados:

            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                print("[foto " + str(amostra) + " capturada com sucesso]")
                amostra += 1

    cv2.imshow('Frame', imagem)
    cv2.waitKey(1)
    if(amostra>= numeroAmostra + 1):
        break
print("Faces capturadas com sucesso!")
camera.release()
cv2.destroyAllWindows()
