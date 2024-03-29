#include <fstream>
#include <thread>
#include <random>
#include <time.h>
#include <Windows.h>
#include <iostream>

using namespace std;

struct neuron {
	double value;
	double error;
	void act() {
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

class network {
public:
    // Количество слоев в нейронной сети
    int layers;

    // Массив указателей на слои нейронов
    neuron** neurons;

    // Трехмерный массив для хранения весов между нейронами
    double*** weights;

    // Массив размеров каждого слоя
    int* size;

    // Количество потоков
    int threadsNum;

    // Деструктор для освобождения выделенной памяти
    ~network() {
        // Освобождение памяти для массива нейронов
        for (int i = 0; i < layers; i++) {
            delete[] neurons[i];
        }
        delete[] neurons;

        // Освобождение памяти для массива весов
        for (int i = 0; i < layers - 1; i++) {
            for (int j = 0; j < size[i]; j++) {
                delete[] weights[i][j];
            }
            delete[] weights[i];
        }
        delete[] weights;

        // Освобождение памяти для массива размеров слоев
        delete[] size;
    }

    // Установка параметров сети без обучения из файла
    void setLayersNotStudy(int n, int* p, string filename)
    {
        ifstream fin;
        fin.open(filename);
        srand(time(0));
        layers = n;
        neurons = new neuron * [n];
        weights = new double** [n - 1];
        size = new int[n];
        for (int i = 0; i < n; i++)
        {
            size[i] = p[i];
            neurons[i] = new neuron[p[i]];
            if (i < n - 1)
            {
                weights[i] = new double* [p[i]];
                for (int j = 0; j < p[i]; j++)
                {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k = 0; k < p[i + 1]; k++)
                    {
                        fin >> weights[i][j][k];
                    }
                }
            }
        }
    }

    // Функция для вычисления производной сигмоидной функции
    double sigm_pro(double x)
    {
        if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
        double res = x * (1.0 - x);
        return res;
    }

    // Простая функция предсказания
    double predict(double x)
    {
        if (x >= 0.8)
            return 1;
        else
            return 0;
    }

    // Установка параметров сети и инициализация случайными весами
    void setLayers(int n, int* p)
    {
        srand(time(0));
        layers = n;
        neurons = new neuron * [n];
        weights = new double** [n - 1];
        size = new int[n];
        for (int i(0); i < n; i++)
        {
            size[i] = p[i];
            neurons[i] = new neuron[p[i]];
            if (i < n - 1)
            {
                weights[i] = new double* [p[i]];
                for (int j(0); j < p[i]; j++)
                {
                    weights[i][j] = new double[p[i + 1]];
                    for (int k(0); k < p[i + 1]; k++)
                    {
                        // Инициализация весов случайными значениями
                        weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];
                    }
                }
            }
        }
    }

    // Установка входных значений
    void set_input(double* p)
    {
        for (int i = 0; i < size[0]; i++)
        {
            neurons[0][i].value = p[i];
        }
    }

    // Зануление значений нейронов в указанном слое
    void LayersCleaner(int LayerNumber, int start, int stop)
    {
        srand(time(0));
        for (int i = start; i < stop; i++)
        {
            neurons[LayerNumber][i].value = 0;
        }
    }

    // Прямое распространение входных данных по сети
    void ForwardFeeder(int LayerNumber, int start, int stop)
    {
        for (int j = start; j < stop; j++)
        {
            for (int k(0); k < size[LayerNumber - 1]; k++) {
                neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
            }
            neurons[LayerNumber][j].act();
        }
    }

    // Прямое распространение по всей сети
    double ForwardFeed()
    {
        setlocale(LC_ALL, "Russian");

        for (int i = 1; i < layers; i++)
        {
            // Зануление или обнуление значений нейронов в текущем слое
            LayersCleaner(i, 0, size[i]);

            // Прямое распространение для передачи значений нейронов
            ForwardFeeder(i, 0, size[i]);
        }

        double max = 0;
        double prediction = 0;

        // Поиск максимального значения на выходном слое
        for (int i = 0; i < size[layers - 1]; i++)
        {
            if (neurons[layers - 1][i].value > max)
            {
                max = neurons[layers - 1][i].value;
                prediction = i;
            }
        }

        return prediction;
    }

    // Прямое распространение с возможностью передачи номера темы
    double ForwardFeed(int tem)
    {
        setlocale(LC_ALL, "Russian");

        double max = 0;
        double prediction = 0;

        for (int i = 1; i < layers; i++)
            ForwardFeeder(i, 0, size[i]);

        // Поиск максимального значения на выходном слое
        for (int i = 0; i < size[layers - 1]; i++)
        {
            if (neurons[layers - 1][i].value > max)
            {
                max = neurons[layers - 1][i].value;
                prediction = i;
            }
        }

        return prediction;
    }

    // Расчет ошибки для каждого нейрона в указанном слое
    void ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr)
    {
        if (LayerNumber == layers - 1)
        {
            for (int j = start; j < stop; j++)
            {
                if (j != int(rresult))
                {
                    neurons[LayerNumber][j].error = -(neurons[LayerNumber][j].value);
                }
                else
                {
                    neurons[LayerNumber][j].error = 1.0 - (neurons[LayerNumber][j].value);
                }
            }
        }
        else
        {
            for (int j = start; j < stop; j++)
            {
                double error = 0.0;
                for (int k = 0; k < size[LayerNumber + 1]; k++)
                {
                    error += neurons[LayerNumber + 1][k].error * weights[LayerNumber][j][k];
                }
                neurons[LayerNumber][j].error = error;
            }
        }
    }

    // Обновление весов на указанном слое
    void WeightsUpdater(int start, int stop, int LayerNum, int lr)
    {
        int i = LayerNum;
        for (int j = start; j < stop; j++)
        {
            for (int k = 0; k < size[i + 1]; k++)
            {
                weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
            }
        }
    }

    // Обратное распространение ошибки по сети
    void BackPropogation(double prediction, double rresult, double lr)
    {
        for (int i = layers - 1; i > 0; i--)
        {
            // Вычисление ошибки на выходном слое
            if (i == layers - 1)
            {
                for (int j = 0; j < size[i]; j++)
                {
                    if (j != int(rresult))
                        neurons[i][j].error = -(neurons[i][j].value);
                    else
                        neurons[i][j].error = 1.0 - (neurons[i][j].value);
                }
            }
            else
            {
                for (int j = 0; j < size[i]; j++)
                {
                    double error = 0.0;
                    for (int k = 0; k < size[i + 1]; k++)
                    {
                        error += neurons[i + 1][k].error * weights[i][j][k];
                    }
                    neurons[i][j].error = error;
                }
            }
        }

        for (int i = 0; i < layers - 1; i++)
        {
            // Обновление весов
            for (int j = 0; j < size[i]; j++)
            {
                for (int k = 0; k < size[i + 1]; k++)
                {
                    weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_pro(neurons[i + 1][k].value) * neurons[i][j].value;
                }
            }
        }
    }

    // Сохранение весов в файл
    bool SaveWeights()
    {
        ofstream fout;
        fout.open("lib/weights.txt");
        for (int i = 0; i < layers; i++)
        {
            if (i < layers - 1)
            {
                for (int j = 0; j < size[i]; j++)
                {
                    for (int k = 0; k < size[i + 1]; k++)
                        fout << weights[i][j][k] << " ";
                }
            }
        }
        fout.close();
        return 1;
    }
};


int main()
{
    // Инициализация генератора случайных чисел
    srand(time(0));

    // Установка локали для корректного отображения текста
    setlocale(LC_ALL, "Russian");

    ifstream fin;
    ifstream ftin;
    ofstream fout;

    // Количество слоев и размеры каждого слоя
    const int l = 4;
    const int input_l = 4096;
    int size[l] = { input_l, 64, 32, 26 };

    // Создание объекта нейронной сети
    network nn;

    double input[input_l];

    char rresult;
    double result;
    double ra = 0;
    int maxra = 0;
    int maxraepoch = 0;
    const int n = 77;
    bool to_study = 0;
    int colT = 0;

    // Запрос пользователя о начале обучения
    cout << "Начать обучение? (0/1): ";
    cin >> to_study;

    double time = 0;
    if (to_study)
    {
        // Установка параметров и случайной инициализации весов для обучения
        nn.setLayers(l, size);
        for (int e = 0; ra / n * 100 < 100; e++)
        {
            fout.open("lib/output.txt");
            if (!fout.is_open()) {
                cout << "Файл не был открыт\nlib/output.txt" << endl;
                return 1;
            }
            cout << "Epoch #" << e << endl;
            double epoch_start = clock();
            ra = 0;
            double w_delta = 0;

            fin.open("lib/lib.txt");

            // Обработка каждого образца из обучающего набора
            for (int i = 0; i < n; i++)
            {
                double start = clock();
                for (int j = 0; j < input_l; j++)
                    fin >> input[j];
                fin >> rresult;
                double stop = clock();
                time += stop - start;
                rresult -= 65;

                // Установка входных значений
                nn.set_input(input);

                // Прямое распространение
                result = nn.ForwardFeed();

                // Проверка правильности предсказания
                if (result == rresult)
                {
                    cout << "Определена буква: " << char(rresult + 65) << "\t\t\t****" << endl;
                    ra++;
                }
                else
                {
                    // Обратное распространение ошибки в случае ошибки предсказания
                    nn.BackPropogation(result, rresult, 0.6);
                }
            }
            fin.close();
            double epoch_stop = clock();
            cout << "Правильные ответы: " << ra / n * 100 << "% \t Максимальная правильность: " << double(maxra) / n * 100 << "(эпоха " << maxraepoch << " )" << endl;

            cout << "Время, необходимое для завершения: " << time / 1000 << " мс\t\t\tВремя эпохи: " << epoch_stop - epoch_start << endl;
            time = 0;

            if (ra > maxra)
            {
                maxra = ra;
                maxraepoch = e;
            }
            if (maxraepoch < e - 250)
            {
                maxra = 0;
            }
        }
        if (nn.SaveWeights())
        {
            cout << "Веса успешно сохранены!";
        }
    }
    else
    {
        // Загрузка весов из файла, если обучение не требуется
        nn.setLayersNotStudy(l, size, "lib/perfect_weights.txt");
    }
    fin.close();

    // Запуск тестирования
    cout << "Начать тест?:(1/0) ";
    bool to_start_test = 0;
    cin >> to_start_test;
    if (to_start_test)
    {
        cout << "Количество примеров в тесте: ";
        cin >> colT;
        ftin.open("lib/test.txt");
        if (ftin.is_open())
        {
            for (int j(0); j < colT; j++)
            {
                for (int i = 0; i < input_l; i++)
                    ftin >> input[i];
                ftin >> rresult;
                cout << "По факту буква: " << rresult << endl;
                nn.set_input(input);
                result = nn.ForwardFeed();
                cout << "Определено как буква: " << char(result + 65) << "\n\n";
            }
        }
    }
    ftin.close();
    cin >> rresult;
    return 0;
}