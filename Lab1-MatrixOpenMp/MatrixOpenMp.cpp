#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace std::chrono;

const char* OUTPUT_FILE_NAME_1 = "MatrixOpenMp.output";

struct Matrix {
    int rows, columns;
    int** values;
};

Matrix toMatrix(int rows, int columns, int** values) {
    Matrix result;
    result.rows = rows;
    result.columns = columns;
    result.values = values;
    return result;
}

Matrix readMatrix(char* file) {
    ifstream stream(file);
    int rows, columns;
    stream >> rows >> columns;
    int** values = new int* [rows];
    for (int row = 0; row < rows; row++) {
        values[row] = new int[columns];
        for (int column = 0; column < columns; column++) {
            stream >> values[row][column];
        }
    }
    stream.close();
    return toMatrix(rows, columns, values);
}

Matrix generateMatrix(int rows, int columns) {
    int** values = new int* [rows];
    for (int row = 0; row < rows; row++) {
        values[row] = new int[columns];
        for (int column = 0; column < columns; column++) {
            values[row][column] = -10 + rand() % 20;
        }
    }
    return toMatrix(rows, columns, values);
}

void printMatrix(Matrix matrix) {
    ofstream stream(OUTPUT_FILE_NAME_1);
    stream << matrix.rows << " " << matrix.columns << endl;
    for (int row = 0; row < matrix.rows; row++) {
        for (int column = 0; column < matrix.columns; column++) {
            stream << matrix.values[row][column] << " ";
        }
        stream << endl;
    }
    stream.close();
}

int calculateValue(Matrix x, Matrix y, int row, int column) {
    int result = 0;
    for (int i = 0; i < x.columns; i++) {
        result += x.values[row][i] * y.values[i][column];
    }
    return result;
}

Matrix serialMatrixMultiply(Matrix x, Matrix y) {
    int rows = x.rows;
    int columns = y.columns;
    int** result = new int* [rows];
    for (int row = 0; row < rows; row++) {
        result[row] = new int[columns];
        for (int column = 0; column < columns; column++) {
            result[row][column] = calculateValue(x, y, row, column);
        }
    }
    return toMatrix(rows, columns, result);
}
Matrix parallelSimpleMatrixMultiply(Matrix x, Matrix y) {
    int rows = x.rows;
    int columns = y.columns;
    int** result = new int* [rows];

#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        result[row] = new int[columns];
        for (int column = 0; column < columns; column++) {
            result[row][column] = calculateValue(x, y, row, column);
        }
    }
    return toMatrix(rows, columns, result);
}
Matrix parallelTaskStaticMatrixMultiply(Matrix x, Matrix y) {
    int rows = x.rows;
    int columns = y.columns;

    int** result = new int* [rows];
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        result[row] = new int[columns];
    }

    int tasks = rows * columns;
#pragma omp parallel for schedule(static)
    for (int task = 0; task < tasks; task++) {
        int row = task / columns;
        int column = task % columns;
        result[row][column] = calculateValue(x, y, row, column);
    }

    return toMatrix(rows, columns, result);
}
Matrix parallelTaskDynamicMatrixMultiply(Matrix x, Matrix y) {
    int rows = x.rows;
    int columns = y.columns;

    int** result = new int* [rows];
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        result[row] = new int[columns];
    }

    int tasks = rows * columns;
#pragma omp parallel for schedule(dynamic, 1000)
    for (int task = 0; task < tasks; task++) {
        int row = task / columns;
        int column = task % columns;
        result[row][column] = calculateValue(x, y, row, column);
    }

    return toMatrix(rows, columns, result);
}
Matrix parallelTaskGuidedMatrixMultiply(Matrix x, Matrix y) {
    int rows = x.rows;
    int columns = y.columns;

    int** result = new int* [rows];
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        result[row] = new int[columns];
    }

    int tasks = rows * columns;
#pragma omp parallel for schedule(guided, 1000)
    for (int task = 0; task < tasks; task++) {
        int row = task / columns;
        int column = task % columns;
        result[row][column] = calculateValue(x, y, row, column);
    }

    return toMatrix(rows, columns, result);
}

Matrix measureMatrixMultiply(Matrix x, Matrix y, int mode) {
    const char* modeName = "";
    Matrix result;
    auto start = high_resolution_clock::now();
    switch (mode) {
    case 0:
        result = serialMatrixMultiply(x, y);
        modeName = "serialMatrixMultiply";
        break;
    case 1:
        result = parallelSimpleMatrixMultiply(x, y);
        modeName = "parallelSimpleMatrixMultiply";
        break;
    case 2:
        result = parallelTaskStaticMatrixMultiply(x, y);
        modeName = "parallelTaskStaticMatrixMultiply";
        break;
    case 3:
        result = parallelTaskDynamicMatrixMultiply(x, y);
        modeName = "parallelTaskDynamicMatrixMultiply";
        break;
    case 4:
        result = parallelTaskGuidedMatrixMultiply(x, y);
        modeName = "parallelTaskGuidedMatrixMultiply";
        break;
    default:
        break;
    }
    auto finish = high_resolution_clock::now();
    cout << "For mode " << modeName << ": " << duration<double>(finish - start).count() << endl;
    return result;
}

void measureMatrixMultiplyModes(Matrix x, Matrix y){
    cout << "Running in all modes, sizes: rows = " << x.rows << ", intermediate = " << x.columns << ", columns = " << y.columns << endl;
    Matrix result;
    for (int i = 0; i < 5; i++) {
        result = measureMatrixMultiply(x, y, i);
    }
    printMatrix(result);
}

int main1(int argc, char* argv[]) {
    if (argc == 3) {
        // read file mode
        measureMatrixMultiplyModes(readMatrix(argv[1]), readMatrix(argv[2]));
    }
    else {
        // random generated values mode
        // square
        measureMatrixMultiplyModes(generateMatrix(100, 100), generateMatrix(100, 100));
        measureMatrixMultiplyModes(generateMatrix(500, 500), generateMatrix(500, 500));
        measureMatrixMultiplyModes(generateMatrix(800, 800), generateMatrix(800, 800));

        // rectangle
        measureMatrixMultiplyModes(generateMatrix(100, 200), generateMatrix(200, 250));
        measureMatrixMultiplyModes(generateMatrix(500, 400), generateMatrix(400, 200));
        measureMatrixMultiplyModes(generateMatrix(300, 600), generateMatrix(600, 750));

        // vector
        measureMatrixMultiplyModes(generateMatrix(1, 300), generateMatrix(300, 50000));
        measureMatrixMultiplyModes(generateMatrix(50000, 300), generateMatrix(300, 1));
    }
    return 0;
}