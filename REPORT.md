# Отчет по лабораторной работе по курсу "Искусственный интеллект"
---

| Студент | *Марков А.Н.* |
|------|------|
| Группа  | *М8О-308Б-18* |

В ходе данной лабораторной работы решалась задача классификации для датасета MNIST с помощью 1, 2 и 3-слойного перспетрона.

## Часть 1. Создание персептрона на основе базовых операций библиотеки numpy
---

За основу был взят код из лекции. Я пытался реализовать программу в виде библиотеки, которая могла бы быть использована для широкого круга задач.
В ней "кирпичики" нейронных сетей (линейное преобразование, функции активации, функции ошибок) вынесены в отдельные классы. Эти "кирпичики" являются слоями в нейронной сети. Нейронная сети представлена классом Net.

Были реализованы персептроны с 1, 2 и 3 слоями.

### Результаты для однослойного персептрона
---

```
Начальная точность на тестовой выборке: 0.102024
Конечная точность на тестовой выборке: 0.923571
Матрица неточности:
```

![alt](images/res_1_perc_myfw.png)

![alt](images/res_1_perc_plot_myfw.png)

### Результаты для двухслойного персептрона
---

```
Начальная точность на тестовой выборке: 0.067381
Конечная точность на тестовой выборке: 0.974524
Матрица неточности:
```

![alt](images/res_2_perc_myfw.png)

![alt](images/res_2_perc_plot_myfw.png)

### Результаты для трехслойного персептрона
---

```
Начальная точность на тестовой выборке: 0.106548
Конечная точность на тестовой выборке: 0.975833
Матрица неточности:
```

![alt](images/res_3_perc_myfw.png)

![alt](images/res_3_perc_plot_myfw.png)


### Сравнение передаточных функций
---

Было проведено сравнение различных функций активации. Рассматривались следующие функции активации:

* Тождественная функция 

* Сигмоида 

* Полулинейная функция 

* Гиперболический тангенс

Сравнение проводилось для двухслойного персептрона.

Результаты:

1. Гиперболический тангенс

```
Начальная точность Tanh на тестовой выборке: 0.078571
Конечная точность Tanh на тестовой выборке: 0.972619
```

![alt](images/tanh_myfw.png)

![alt](images/tanh_plot_myfw.png)

2. Тождественная функция

```
Начальная точность Identity на тестовой выборке: 0.100833
Конечная точность Identity на тестовой выборке: 0.917619
```

![alt](images/identity_myfw.png)

![alt](images/identity_plot_myfw.png)

3. Сигмоида

```
Начальная точность Sigmoid на тестовой выборке: 0.096905
Конечная точность Sigmoid на тестовой выборке: 0.963810
```

![alt](images/sigmoid_myfw.png)

![alt](images/sigmoid_plot_myfw.png)

4. Полулинейная функция

```
Начальная точность ReLU на тестовой выборке: 0.058810
Конечная точность ReLU на тестовой выборке: 0.974762
```

![alt](images/relu_myfw.png)

![alt](images/relu_plot_myfw.png)

### Сравнение количества нейронов в промежуточных слоях
---

Исследуется двухслойный перспетрон, в котором промежуточная функция - гиперболический тангенс, функция ошибок - cross entropy loss. Количество эпох 15.

```
Точность на тестовой выборке при 2 нейронах в промежуточном слое: 0.388452
Точность на тестовой выборке при 6 нейронах в промежуточном слое: 0.883571
Точность на тестовой выборке при 10 нейронах в промежуточном слое: 0.908095
Точность на тестовой выборке при 14 нейронах в промежуточном слое: 0.928690
Точность на тестовой выборке при 18 нейронах в промежуточном слое: 0.925119
Точность на тестовой выборке при 22 нейронах в промежуточном слое: 0.941190
Точность на тестовой выборке при 26 нейронах в промежуточном слое: 0.940476
Точность на тестовой выборке при 30 нейронах в промежуточном слое: 0.948452
Точность на тестовой выборке при 34 нейронах в промежуточном слое: 0.951071
Точность на тестовой выборке при 38 нейронах в промежуточном слое: 0.957381
Точность на тестовой выборке при 42 нейронах в промежуточном слое: 0.952500
Точность на тестовой выборке при 46 нейронах в промежуточном слое: 0.962262
Точность на тестовой выборке при 50 нейронах в промежуточном слое: 0.955714
Точность на тестовой выборке при 54 нейронах в промежуточном слое: 0.963214
Точность на тестовой выборке при 58 нейронах в промежуточном слое: 0.959405
Точность на тестовой выборке при 62 нейронах в промежуточном слое: 0.964524
Точность на тестовой выборке при 66 нейронах в промежуточном слое: 0.965119
Точность на тестовой выборке при 70 нейронах в промежуточном слое: 0.961190
Точность на тестовой выборке при 74 нейронах в промежуточном слое: 0.969405
Точность на тестовой выборке при 78 нейронах в промежуточном слое: 0.965357
Точность на тестовой выборке при 82 нейронах в промежуточном слое: 0.968571
Точность на тестовой выборке при 86 нейронах в промежуточном слое: 0.972024
Точность на тестовой выборке при 90 нейронах в промежуточном слое: 0.971310
Точность на тестовой выборке при 94 нейронах в промежуточном слое: 0.970000
Точность на тестовой выборке при 98 нейронах в промежуточном слое: 0.971429
Точность на тестовой выборке при 102 нейронах в промежуточном слое: 0.971190
Точность на тестовой выборке при 106 нейронах в промежуточном слое: 0.971667
Точность на тестовой выборке при 110 нейронах в промежуточном слое: 0.972857
Точность на тестовой выборке при 114 нейронах в промежуточном слое: 0.972500
Точность на тестовой выборке при 118 нейронах в промежуточном слое: 0.971667
Точность на тестовой выборке при 122 нейронах в промежуточном слое: 0.972143
Точность на тестовой выборке при 126 нейронах в промежуточном слое: 0.975238
Точность на тестовой выборке при 130 нейронах в промежуточном слое: 0.973929
Точность на тестовой выборке при 134 нейронах в промежуточном слое: 0.972619
Точность на тестовой выборке при 138 нейронах в промежуточном слое: 0.973452
Точность на тестовой выборке при 142 нейронах в промежуточном слое: 0.972619
Точность на тестовой выборке при 146 нейронах в промежуточном слое: 0.972024
Точность на тестовой выборке при 150 нейронах в промежуточном слое: 0.976310
Точность на тестовой выборке при 154 нейронах в промежуточном слое: 0.973333
Точность на тестовой выборке при 158 нейронах в промежуточном слое: 0.975833
Точность на тестовой выборке при 162 нейронах в промежуточном слое: 0.973452
Точность на тестовой выборке при 166 нейронах в промежуточном слое: 0.969643
Точность на тестовой выборке при 170 нейронах в промежуточном слое: 0.975000
Точность на тестовой выборке при 174 нейронах в промежуточном слое: 0.974405
Точность на тестовой выборке при 178 нейронах в промежуточном слое: 0.974643
Точность на тестовой выборке при 182 нейронах в промежуточном слое: 0.974881
Точность на тестовой выборке при 186 нейронах в промежуточном слое: 0.970595
Точность на тестовой выборке при 190 нейронах в промежуточном слое: 0.972976
Точность на тестовой выборке при 194 нейронах в промежуточном слое: 0.974286
Точность на тестовой выборке при 198 нейронах в промежуточном слое: 0.969405
Точность на тестовой выборке при 202 нейронах в промежуточном слое: 0.973452
Точность на тестовой выборке при 206 нейронах в промежуточном слое: 0.972738
Точность на тестовой выборке при 210 нейронах в промежуточном слое: 0.971190
Точность на тестовой выборке при 214 нейронах в промежуточном слое: 0.972619
Точность на тестовой выборке при 218 нейронах в промежуточном слое: 0.974286
Точность на тестовой выборке при 222 нейронах в промежуточном слое: 0.973571
Точность на тестовой выборке при 226 нейронах в промежуточном слое: 0.975119
Точность на тестовой выборке при 230 нейронах в промежуточном слое: 0.967024
Точность на тестовой выборке при 234 нейронах в промежуточном слое: 0.969881
Точность на тестовой выборке при 238 нейронах в промежуточном слое: 0.974881
Точность на тестовой выборке при 242 нейронах в промежуточном слое: 0.972976
Точность на тестовой выборке при 246 нейронах в промежуточном слое: 0.972738
Точность на тестовой выборке при 250 нейронах в промежуточном слое: 0.973095
Точность на тестовой выборке при 254 нейронах в промежуточном слое: 0.975476
``` 

## Часть 2. Создание персептрона на основе фреймоврка pytorch
---

### Результаты для однослойного персептрона

```
Epoch 1. Training loss = 0.583537. Training accuracy = 0.870149
Epoch 2. Training loss = 0.566510. Training accuracy = 0.890893
Epoch 3. Training loss = 0.555069. Training accuracy = 0.896339
Epoch 4. Training loss = 0.545561. Training accuracy = 0.899256
Epoch 5. Training loss = 0.541058. Training accuracy = 0.901875
Epoch 6. Training loss = 0.537418. Training accuracy = 0.902857
Epoch 7. Training loss = 0.536963. Training accuracy = 0.904643
Epoch 8. Training loss = 0.531272. Training accuracy = 0.906012
Epoch 9. Training loss = 0.528730. Training accuracy = 0.906369
Epoch 10. Training loss = 0.525982. Training accuracy = 0.907202
Epoch 11. Training loss = 0.521922. Training accuracy = 0.908839
Epoch 12. Training loss = 0.519704. Training accuracy = 0.909702
Epoch 13. Training loss = 0.519917. Training accuracy = 0.909524
Epoch 14. Training loss = 0.514948. Training accuracy = 0.911071
Epoch 15. Training loss = 0.511764. Training accuracy = 0.911518
Test loss = 0.879129. Test accuracy = 0.894286
```

![alt](images/res_1_perc_torch.png)

![alt](images/res_1_perc_plot_torch.png)

### Результаты для двухслойного персептрона

```
Epoch 1. Training loss = 0.425004. Training accuracy = 0.880714
Epoch 2. Training loss = 0.336434. Training accuracy = 0.914673
Epoch 3. Training loss = 0.324468. Training accuracy = 0.919940
Epoch 4. Training loss = 0.317552. Training accuracy = 0.925357
Epoch 5. Training loss = 0.295970. Training accuracy = 0.927857
Epoch 6. Training loss = 0.291290. Training accuracy = 0.931071
Epoch 7. Training loss = 0.280326. Training accuracy = 0.933899
Epoch 8. Training loss = 0.276879. Training accuracy = 0.935208
Epoch 9. Training loss = 0.272811. Training accuracy = 0.939107
Epoch 10. Training loss = 0.270931. Training accuracy = 0.938482
Epoch 11. Training loss = 0.259682. Training accuracy = 0.940327
Epoch 12. Training loss = 0.248170. Training accuracy = 0.943363
Epoch 13. Training loss = 0.259900. Training accuracy = 0.942560
Epoch 14. Training loss = 0.260432. Training accuracy = 0.942827
Epoch 15. Training loss = 0.249646. Training accuracy = 0.944494
Test loss = 0.743546. Test accuracy = 0.933690
```

![alt](images/res_2_perc_torch.png)

![alt](images/res_2_perc_plot_torch.png)

### Результаты для трехслойного персептрона

```
Epoch 1. Training loss = 0.498121. Training accuracy = 0.861786
Epoch 2. Training loss = 0.377700. Training accuracy = 0.906548
Epoch 3. Training loss = 0.335700. Training accuracy = 0.917738
Epoch 4. Training loss = 0.321292. Training accuracy = 0.924018
Epoch 5. Training loss = 0.302991. Training accuracy = 0.928185
Epoch 6. Training loss = 0.318627. Training accuracy = 0.929970
Epoch 7. Training loss = 0.278445. Training accuracy = 0.934167
Epoch 8. Training loss = 0.296303. Training accuracy = 0.933214
Epoch 9. Training loss = 0.281153. Training accuracy = 0.937232
Epoch 10. Training loss = 0.275140. Training accuracy = 0.938810
Epoch 11. Training loss = 0.283443. Training accuracy = 0.936369
Epoch 12. Training loss = 0.287795. Training accuracy = 0.938333
Epoch 13. Training loss = 0.282095. Training accuracy = 0.938095
Epoch 14. Training loss = 0.273779. Training accuracy = 0.939196
Epoch 15. Training loss = 0.281412. Training accuracy = 0.938423
Test loss = 0.531065. Test accuracy = 0.938690
```

![alt](images/res_3_perc_torch.png)

![alt](images/res_3_perc_plot_torch.png)

### Сравнение пердаточных функций
---

Сравнивались relu, tanh, sigmoid.

1. relu

```
ReLU()
Epoch 1. Training loss = 0.419553. Training accuracy = 0.882083
Epoch 2. Training loss = 0.336638. Training accuracy = 0.914970
Epoch 3. Training loss = 0.315929. Training accuracy = 0.922292
Epoch 4. Training loss = 0.296976. Training accuracy = 0.928065
Epoch 5. Training loss = 0.288788. Training accuracy = 0.931935
Epoch 6. Training loss = 0.276954. Training accuracy = 0.933006
Epoch 7. Training loss = 0.275378. Training accuracy = 0.935655
Epoch 8. Training loss = 0.280434. Training accuracy = 0.935774
Epoch 9. Training loss = 0.260995. Training accuracy = 0.940417
Epoch 10. Training loss = 0.258672. Training accuracy = 0.939702
Epoch 11. Training loss = 0.247163. Training accuracy = 0.941696
Epoch 12. Training loss = 0.248355. Training accuracy = 0.941220
Epoch 13. Training loss = 0.256135. Training accuracy = 0.944256
Epoch 14. Training loss = 0.240635. Training accuracy = 0.944702
Epoch 15. Training loss = 0.235668. Training accuracy = 0.947827
Test loss = 0.592308. Test accuracy = 0.930476
```

![alt](images/relu_torch.png)

![alt](images/relu_plot_torch.png)

2. tanh

```
Tanh()
Epoch 1. Training loss = 0.602542. Training accuracy = 0.839077
Epoch 2. Training loss = 0.528208. Training accuracy = 0.874732
Epoch 3. Training loss = 0.492965. Training accuracy = 0.888690
Epoch 4. Training loss = 0.460490. Training accuracy = 0.900208
Epoch 5. Training loss = 0.457080. Training accuracy = 0.903482
Epoch 6. Training loss = 0.433638. Training accuracy = 0.910000
Epoch 7. Training loss = 0.434876. Training accuracy = 0.911607
Epoch 8. Training loss = 0.423171. Training accuracy = 0.918631
Epoch 9. Training loss = 0.411043. Training accuracy = 0.917530
Epoch 10. Training loss = 0.396182. Training accuracy = 0.922917
Epoch 11. Training loss = 0.376775. Training accuracy = 0.927232
Epoch 12. Training loss = 0.387205. Training accuracy = 0.927262
Epoch 13. Training loss = 0.379855. Training accuracy = 0.927827
Epoch 14. Training loss = 0.373806. Training accuracy = 0.931339
Epoch 15. Training loss = 0.377989. Training accuracy = 0.930268
Test loss = 0.371834. Test accuracy = 0.935595
```

![alt](images/tanh_torch.png)

![alt](images/tanh_plot_torch.png)

3. sigmoid

```
Sigmoid()
Epoch 1. Training loss = 0.319196. Training accuracy = 0.902887
Epoch 2. Training loss = 0.223063. Training accuracy = 0.935744
Epoch 3. Training loss = 0.201841. Training accuracy = 0.941012
Epoch 4. Training loss = 0.192190. Training accuracy = 0.946696
Epoch 5. Training loss = 0.187241. Training accuracy = 0.949941
Epoch 6. Training loss = 0.176700. Training accuracy = 0.951190
Epoch 7. Training loss = 0.167425. Training accuracy = 0.955060
Epoch 8. Training loss = 0.161521. Training accuracy = 0.956637
Epoch 9. Training loss = 0.148013. Training accuracy = 0.961994
Epoch 10. Training loss = 0.146608. Training accuracy = 0.960298
Epoch 11. Training loss = 0.136815. Training accuracy = 0.964256
Epoch 12. Training loss = 0.135681. Training accuracy = 0.964732
Epoch 13. Training loss = 0.136262. Training accuracy = 0.965327
Epoch 14. Training loss = 0.134478. Training accuracy = 0.966696
Epoch 15. Training loss = 0.126380. Training accuracy = 0.968155
Test loss = 0.205653. Test accuracy = 0.955595
```

![alt](images/sigmoid_torch.png)

![alt](images/sigmoid_plot_torch.png)

### Сравнение количества нейронов в промежуточных слоях
---

Исследуется двухслойный перспетрон, в котором промежуточная функция - ReLU, функция ошибок - cross entropy loss. Количество эпох 15.

```
Количество нейронов в промежуточном слое = 2;
Epoch 1. Training loss = 1.742130. Training accuracy = 0.275417
Epoch 2. Training loss = 1.650457. Training accuracy = 0.330774
Epoch 3. Training loss = 1.622662. Training accuracy = 0.345804
Epoch 4. Training loss = 1.617515. Training accuracy = 0.346905
Epoch 5. Training loss = 1.615442. Training accuracy = 0.349375
Epoch 6. Training loss = 1.614044. Training accuracy = 0.350179
Epoch 7. Training loss = 1.612765. Training accuracy = 0.351161
Epoch 8. Training loss = 1.612316. Training accuracy = 0.351696
Epoch 9. Training loss = 1.611894. Training accuracy = 0.352649
Epoch 10. Training loss = 1.611498. Training accuracy = 0.353452
Epoch 11. Training loss = 1.611096. Training accuracy = 0.354435
Epoch 12. Training loss = 1.610723. Training accuracy = 0.354702
Epoch 13. Training loss = 1.610450. Training accuracy = 0.355208
Epoch 14. Training loss = 1.610287. Training accuracy = 0.354881
Epoch 15. Training loss = 1.610205. Training accuracy = 0.355357
Test loss = 1.634992. Test accuracy = 0.342738
Количество нейронов в промежуточном слое = 52;
Epoch 1. Training loss = 0.433501. Training accuracy = 0.877470
Epoch 2. Training loss = 0.334820. Training accuracy = 0.915268
Epoch 3. Training loss = 0.314838. Training accuracy = 0.920714
Epoch 4. Training loss = 0.298818. Training accuracy = 0.924345
Epoch 5. Training loss = 0.289761. Training accuracy = 0.928720
Epoch 6. Training loss = 0.283988. Training accuracy = 0.930179
Epoch 7. Training loss = 0.282090. Training accuracy = 0.932351
Epoch 8. Training loss = 0.272902. Training accuracy = 0.934226
Epoch 9. Training loss = 0.264595. Training accuracy = 0.935714
Epoch 10. Training loss = 0.267743. Training accuracy = 0.936816
Epoch 11. Training loss = 0.267193. Training accuracy = 0.935119
Epoch 12. Training loss = 0.254028. Training accuracy = 0.939554
Epoch 13. Training loss = 0.258932. Training accuracy = 0.937143
Epoch 14. Training loss = 0.255647. Training accuracy = 0.939881
Epoch 15. Training loss = 0.246441. Training accuracy = 0.941607
Test loss = 0.447754. Test accuracy = 0.923810
Количество нейронов в промежуточном слое = 102;
Epoch 1. Training loss = 0.435009. Training accuracy = 0.877321
Epoch 2. Training loss = 0.345285. Training accuracy = 0.909762
Epoch 3. Training loss = 0.331445. Training accuracy = 0.916042
Epoch 4. Training loss = 0.308537. Training accuracy = 0.922024
Epoch 5. Training loss = 0.303626. Training accuracy = 0.924524
Epoch 6. Training loss = 0.296322. Training accuracy = 0.927798
Epoch 7. Training loss = 0.286435. Training accuracy = 0.929345
Epoch 8. Training loss = 0.289999. Training accuracy = 0.932946
Epoch 9. Training loss = 0.269294. Training accuracy = 0.935298
Epoch 10. Training loss = 0.268446. Training accuracy = 0.935685
Epoch 11. Training loss = 0.267261. Training accuracy = 0.936994
Epoch 12. Training loss = 0.258517. Training accuracy = 0.939315
Epoch 13. Training loss = 0.264103. Training accuracy = 0.938958
Epoch 14. Training loss = 0.261155. Training accuracy = 0.939613
Epoch 15. Training loss = 0.260345. Training accuracy = 0.939494
Test loss = 0.583918. Test accuracy = 0.922976
Количество нейронов в промежуточном слое = 152;
Epoch 1. Training loss = 0.427557. Training accuracy = 0.879643
Epoch 2. Training loss = 0.342218. Training accuracy = 0.911101
Epoch 3. Training loss = 0.325425. Training accuracy = 0.920387
Epoch 4. Training loss = 0.304077. Training accuracy = 0.924702
Epoch 5. Training loss = 0.293346. Training accuracy = 0.927946
Epoch 6. Training loss = 0.288252. Training accuracy = 0.930625
Epoch 7. Training loss = 0.282443. Training accuracy = 0.933155
Epoch 8. Training loss = 0.280774. Training accuracy = 0.934018
Epoch 9. Training loss = 0.275842. Training accuracy = 0.935893
Epoch 10. Training loss = 0.271281. Training accuracy = 0.938214
Epoch 11. Training loss = 0.277610. Training accuracy = 0.938690
Epoch 12. Training loss = 0.263536. Training accuracy = 0.939554
Epoch 13. Training loss = 0.258012. Training accuracy = 0.940119
Epoch 14. Training loss = 0.241322. Training accuracy = 0.944613
Epoch 15. Training loss = 0.264454. Training accuracy = 0.942113
Test loss = 0.665476. Test accuracy = 0.929405
Количество нейронов в промежуточном слое = 202;
Epoch 1. Training loss = 0.420626. Training accuracy = 0.884137
Epoch 2. Training loss = 0.337754. Training accuracy = 0.918244
Epoch 3. Training loss = 0.314710. Training accuracy = 0.925774
Epoch 4. Training loss = 0.303300. Training accuracy = 0.928750
Epoch 5. Training loss = 0.281213. Training accuracy = 0.932202
Epoch 6. Training loss = 0.285897. Training accuracy = 0.933571
Epoch 7. Training loss = 0.270185. Training accuracy = 0.940268
Epoch 8. Training loss = 0.274303. Training accuracy = 0.939851
Epoch 9. Training loss = 0.271272. Training accuracy = 0.941250
Epoch 10. Training loss = 0.248746. Training accuracy = 0.943095
Epoch 11. Training loss = 0.249876. Training accuracy = 0.943988
Epoch 12. Training loss = 0.253441. Training accuracy = 0.945030
Epoch 13. Training loss = 0.244785. Training accuracy = 0.946548
Epoch 14. Training loss = 0.239442. Training accuracy = 0.948274
Epoch 15. Training loss = 0.244356. Training accuracy = 0.946815
Test loss = 0.477109. Test accuracy = 0.938095
Количество нейронов в промежуточном слое = 252;
Epoch 1. Training loss = 0.426117. Training accuracy = 0.880952
Epoch 2. Training loss = 0.338511. Training accuracy = 0.917679
Epoch 3. Training loss = 0.315944. Training accuracy = 0.924673
Epoch 4. Training loss = 0.303782. Training accuracy = 0.929524
Epoch 5. Training loss = 0.274209. Training accuracy = 0.932827
Epoch 6. Training loss = 0.283812. Training accuracy = 0.935952
Epoch 7. Training loss = 0.272913. Training accuracy = 0.938631
Epoch 8. Training loss = 0.269025. Training accuracy = 0.940268
Epoch 9. Training loss = 0.260871. Training accuracy = 0.943095
Epoch 10. Training loss = 0.254953. Training accuracy = 0.942708
Epoch 11. Training loss = 0.244605. Training accuracy = 0.946637
Epoch 12. Training loss = 0.257897. Training accuracy = 0.944970
Epoch 13. Training loss = 0.242090. Training accuracy = 0.945417
Epoch 14. Training loss = 0.239752. Training accuracy = 0.947768
Epoch 15. Training loss = 0.235516. Training accuracy = 0.947827
Test loss = 0.585294. Test accuracy = 0.934643
```
