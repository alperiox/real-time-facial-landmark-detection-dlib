import cv2
import dlib

import sys
import os 

import glob

# landmarkı tespit edip döndüren modeli ilk önce indirmek gerekiyor, indirme linki: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# model bz2 türünde sıkıştırılmış olarak geliyor, bu yüzden
# indirdikten sonra bu scripting çalıştığı klasör içerisine veya herhangi bir yere dosyayı çıkartın ve dosya yolunu "predictor_path" e girin

# landmarked_img değişkeninin ne olduğunu aşağılarda görebilirsiniz
# landmarked_img bana frame işlendikten sonra lazım olduğu için

predictor_path = "indirilen modelin yolunu buraya girin"

# DLIB default HoG face detector modelini kullanacağız, cpuda güzel çalışıyor diye duymuştum
detector = dlib.get_frontal_face_detector()
# indirilen modeli import edelim
predictor = dlib.shape_predictor(predictor_path)

# canlı olarak kullanabilmek için kameraya erişim sağlayalım
video_capture = cv2.VideoCapture(0)

while True:
    # bizim işimiz frame üzerinde olacak, kameradan frameleri çekelim
    ret, frame = video_capture.read()

    # Detectoru kullanarak framedeki yüzlerin bb (bounding box)lerini belirleyelim
    # 1 upsampling için kullanılıyor, sayının büyütülmesi framede daha çok yüzün tanımlanmasını sağlayacaktır
    dets = detector(frame, 1)

    # print("Bulunan yüz sayısı: {}".format(len(dets)))

    for k, d in enumerate(dets):
        # aradığımız bounding boxa "d" üzerinden erişilebiliyor

        # print("Tanımlanan bb no {} ve koordinatlar: Sol: {} Üst: {} Sağ: {} Alt: {}".format(
        #     k, d.left(), d.top(), d.right(), d.bottom()))

        # girdiyi ve bbyi vererek landmarkları alalım
        shape = predictor(frame, d)
        # bb çizilsin
        bb_img = cv2.rectangle(frame, (d.left(), d.top()),
                               (d.right(), d.bottom()), (0, 255, 0), 2)

        # landmarkları istersek teker teker yazdırabiliriz ama gerek yok
        # print("Landmark 0: {}, Landmark 1: {} ...".format(shape.part(0),
        #                                           shape.part(1)))

        # kullanılan landmark modeli 68 adet landmark belirliyor, bunlarıda girdiye çizelim
        for i in range(68):
            # çıktılar dlib ve c++ üzerinden portlandığı için direkt shape üzerinden erişim sağlamak yerine
            # shape.part fonksiyonunu kullanıyoruz, her bir partın .x ve .y attributeları bize x ve y koordinatlarını veriyor
            x = shape.part(i).x
            y = shape.part(i).y
            # landmarkları küçük çemberler halinde çizelim
            landmarked_img = cv2.circle(bb_img, (x, y), 1, (0, 0, 255))
            

        # çıktıları ekrana verelim
        cv2.imshow('Video', landmarked_img)

    # istersek q tuşuna basarak çıkış yapabiliriz
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

