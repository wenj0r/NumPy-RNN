import numpy as np
import numpy.random as r
import scipy.special
import mnist
import cv2
import json
from random import randint

class NeuralNetwork():
    # Показывает текущее изображение
    def show_number(self, img):
        cv2.imshow("Number", img)
        cv2.waitKey(0)

    # Генерирует матрицу значений нейронов, весов, смещений, дельт и ошибок
    def init_nn(self):
        N = {} # Нейроны
        W = {} # Веса
        b = {} # Смещения
        Err = {} # Ошибки
        delta_W = {} # Финальные изменения весов
        temp = {} # Буфер для временных изменений весов

        for i in range(1, len(self.structure)+1):
            N[i] = np.zeros((self.structure[i-1]))
            Err[i] = np.zeros((self.structure[i-1]))
        for i in range(1, len(self.structure)):
            W[i] = r.random_sample((self.structure[i], self.structure[i - 1])) - 0.5
            b[i] = r.random_sample((self.structure[i], 1))
            delta_W[i] = np.zeros((self.structure[i], self.structure[i - 1]))
            temp[i] = np.zeros((self.structure[i], self.structure[i - 1]))

        return N, W, b, Err, delta_W, temp

    # Сигмоида для матрицы
    def sigmoid(self, x):
        return scipy.special.expit(x)

    # Функция квадратичной ошибки для матриц
    def mse(self, x, y):
        # Представление правильного ответа в виде матрицы (например, для двух – это [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        true_answers = np.zeros(10)
        true_answers[y] = 1
        # Подсчет ошибки каждого нейрона выходного слоя
        self.errors[len(self.errors)] = true_answers - x
        # Суммирование квадратичных ошибок нейросети
        self.main_error += sum((true_answers-x) ** 2)/10

    # Прямое распространение
    def linear_activation_forward(self, x):
        self.N[1] = x # инициализация первого слоя
        for layer in range(2, len(self.N)+1):
            layer_input = np.dot(self.W[layer-1], self.N[layer-1]) # умножение матрицы весов и выходов
            layer_output = self.sigmoid(layer_input) # применение функции активации
            self.N[layer] = layer_output
            if layer == len(self.N): # результат
                final_layer = self.N[layer]
        return final_layer

    # Обратное распространение ошибки
    def backpropagation(self, x, y, n):
        # Подсчет ошибки каждого нейрона выходного слоя
        self.mse(x, y)
        # Подсчет ошибки каждого нейрона скрытых слоев
        for layer in range(len(self.errors) - 1, 1, -1):
            self.errors[layer] = np.dot(self.W[layer].T, self.errors[layer + 1])

        # Обновление весов
        for layer in range(len(self.N), 1, -1):
            # Подсчет дельт нейронов
            delta = np.array(self.errors[layer] * self.N[layer] * (1 - self.N[layer]))

            # Подсчет градиентов
            grad = []
            for neuron in self.N[layer-1]:
                grad.append(delta * neuron)

            # Заполнение буфера дельтами весов
            self.temp[layer - 1] += np.array(grad).T

            # Изменения весов слоя в конце пакета
            if (n+1) % self.batch == 0:
                # Изменение весов слоя = градиенты слоя * скорость обучения + момент * изменение в предыдущей итерации
                self.delta_W[layer - 1] = self.temp[layer - 1] * self.learning_speed + self.moment * self.delta_W[layer - 1]
                self.W[layer - 1] += self.delta_W[layer - 1]

        # Очищение буфера для дельт весов
        if (n+1) % self.batch == 0:
            for i in range(1, len(self.structure)):
                self.temp[i] = np.zeros((self.structure[i], self.structure[i - 1]))

    def learning(self):
        for i in range(1, self.epochs+1):
            print('\n------------------------------------------------\nЭпоха {} из {}'.format(i, self.epochs))
            for n in range(0, len(self.x)):
                # Обучение
                answer = self.linear_activation_forward(self.x[n])
                self.backpropagation(answer, self.y[n], n)

                # Вывод ошибки в конце каждого пакета
                if (n+1) % self.batch == 0:
                    loss = round(self.main_error/self.batch*100, 3)
                    print('\rОшибка: {}% --------- пройдено {} из {}'.format(loss, n+1, len(self.x)), end='')
                    self.main_error = 0 # очистка общей ошибки

        print('\n------------------------------------------------\nКонец обучения\n')
        self.main_error = 0  # очистка общей ошибки
        self.evaluate() # проверка на тестовой выборке
    # Тестирование модели
    def evaluate(self):
        for i in range(0, len(self.x_test)):
            answer = self.linear_activation_forward(self.x_test[i])
            self.mse(answer, self.y_test)
        print("Эффективность модели : {}".format(round(100 - (self.main_error / len(self.x_test)), 3)))
        self.main_error = 0

    # Использование модели
    def predict(self, x):
        answer = self.linear_activation_forward(x) # ответ нейросети в виде массива
        number = max(range(len(answer)), key=answer.__getitem__) # ответ, преобразованный в число (индекс максимального элемента массива)
        return number

    # Сохранение модели (тут я использовал json для сохранения массивов)
    def save_model(self, path):
        with open(path, 'w+', encoding='utf-8') as file:
            file.write(json.dumps(self.structure) + '\n')  # запись структуры нейросети
            # Сохранение весов в соответсвии со структурой
            for i in range(1, len(self.structure)):
                file.write(json.dumps(self.W[i].tolist()) + '\n')

    # Загрузка модели (для того, чтобы считывать несколько массивов из одного файла, я сначала считываю их как строку и передаю json парсеру)
    def load_model(self, path):
        with open(path, 'r') as file:
            self.structure = json.loads(file.readline())
            self.N, self.W, self.b, self.errors, self.delta_W, self.temp = self.init_nn()
            for i in range(1, len(self.structure)):
                self.W[i] = np.array(json.loads(file.readline()))

    def __init__(self):
        # Подгрузка обучающих данных, разделенных на две выборки
        # x – содержит в себе все картинки из MNIST, в виде одномерного массива из 784 элементов; а y – правильные ответы
        self.images = mnist.train_images()
        self.input = self.images.reshape((self.images.shape[0], self.images.shape[1] * self.images.shape[2]))

        # Подгрузка обучающей выборки
        self.x = (np.asfarray(self.input[:45000]) / 255.0 * 0.99) + 0.01  # приводит значения к диапазону от 0 до 1
        self.y = mnist.train_labels()[:45000]  # подгрузка правильных ответов

        # Подгрузка тестовой выборки
        self.x_test = (np.asfarray(self.input[45000:]) / 255.0 * 0.99) + 0.01
        self.y_test = mnist.train_labels()[45000:]

        # Гиперпараметры
        self.epochs = 10 # количество полных итераций в обучении
        self.learning_speed = 0.005 # скорость (шаг) градиентного спуска
        self.batch = 100 # размер пакета (веса изменяются в конце пакета); для стохастического обучения нужно установить 1
        self.moment = 0.2 # момент (нужен для преодоления локальных минимумов)
        self.structure = [784, 100, 10]  # структура сети (можно изменять кол-во слоев и нейронов)

        # Инициализация нейросети
        self.main_error = 0 # ошибка в конце каждого пакета
        self.N, self.W, self.b, self.errors, self.delta_W, self.temp = self.init_nn() # матрицы значений нейронов, весов, смещений, дельт и ошибок

# Метод для ручной проверки модели
def test(nn):
    i = randint(0, 14999)
    print('Ответ нейросети – {}'.format(nn.predict(nn.x_test[i])))
    print('Исходное число – {}'.format(nn.y_test[i]))
    nn.show_number(nn.images[45000 + i])

def main():
    nn1 = NeuralNetwork()
    nn1.learning()
    nn1.save_model('model.txt')
    nn1.load_model('model.txt')
    test(nn1)


if __name__ == "__main__":
    main()

### P.S. ###
    # Источники:
    # https://линуксблог.рф/python-nero-seti-chast-1/
    # https://habr.com/ru/post/313216/

    # Пример матрицы
    # W[слой][нейрон, к которому относятся веса][конкретный вес]; слой отсчитываются от единицы
    # [[0.79299055, 0.32493763, 0.0921669 , 0.6863485 , 0.05490081, 0.93221374, 0.03671333, 0.38663013, 0.58974907, 0.71137394],
    #  [0.79376632, 0.24717825, 0.35804966, 0.1163014 , 0.39556726, 0.78367462, 0.36724488, 0.26903711, 0.00252498, 0.61419569],
    #  [0.66623191, 0.70660191, 0.82329327, 0.32646954, 0.004384  , 0.48893273, 0.60011736, 0.16578476, 0.26292606, 0.9343911 ],
    #  [0.08356262, 0.54207804, 0.11726282, 0.30342402, 0.34037449, 0.70535496, 0.60250902, 0.66867626, 0.75693413, 0.96901441]]),
    # [[0.83740363, 0.03620221, 0.05852412, 0.07085195]]