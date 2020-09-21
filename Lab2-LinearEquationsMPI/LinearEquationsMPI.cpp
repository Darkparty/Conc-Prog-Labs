#include <mpi.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

const int PRIMARY_PROCESS = 0;
const char* OUTPUT_FILE_NAME_2 = "LinearEquationsMPI.output";

class LinearEquatSolveProcess {
private:
    int currentProcess;
    int totalProcesses;
    int startPosition;
    int length;

    double precision;
    int m;
    double* A;
    double* b;
    double* x;
    double* xPrev = nullptr;

public:
    LinearEquatSolveProcess(char* coefficientsFile, char* xApproxFile, double precision) {
        this->precision = precision;
        MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
        MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
        if (currentProcess == PRIMARY_PROCESS) {
            readInput(coefficientsFile, xApproxFile);
        }
        shareValues();
    }

    LinearEquatSolveProcess(int m, double precision) {
        this->precision = precision;
        MPI_Comm_rank(MPI_COMM_WORLD, &currentProcess);
        MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
        if (currentProcess == PRIMARY_PROCESS) {
            generateInput(m);
        }
        shareValues();
    }

    ~LinearEquatSolveProcess() {
        delete[] A;
        delete[] b;
        delete[] x;
        delete[] xPrev;
    }

    void shareValues() {
        MPI_Bcast(&m, 1, MPI_INT, PRIMARY_PROCESS, MPI_COMM_WORLD);
        MPI_Bcast(&precision, 1, MPI_DOUBLE, PRIMARY_PROCESS, MPI_COMM_WORLD);

        if (currentProcess != PRIMARY_PROCESS) {
            x = new double[m];
        }
        MPI_Bcast(x, m, MPI_DOUBLE, PRIMARY_PROCESS, MPI_COMM_WORLD);

        if (currentProcess == PRIMARY_PROCESS) {
            int startPosition = 0;
            for (int process = 0; process < totalProcesses; process++) {
                int length = m / totalProcesses + (m % totalProcesses > process);
                if (process == PRIMARY_PROCESS) {
                    this->startPosition = startPosition;
                    this->length = length;
                }
                else {
                    MPI_Send(&startPosition, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
                    MPI_Send(&length, 1, MPI_INT, process, 0, MPI_COMM_WORLD);

                    MPI_Send(&A[startPosition * m], length * m, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
                    MPI_Send(&b[startPosition], length, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
                }
                startPosition += length;
            }
        }
        else {
            MPI_Status status;
            MPI_Recv(&startPosition, 1, MPI_INT, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&length, 1, MPI_INT, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            A = new double[length * m];
            MPI_Recv(A, length * m, MPI_DOUBLE, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            b = new double[length];
            MPI_Recv(b, length, MPI_DOUBLE, PRIMARY_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    void computeValues(int iterationsNumber) {
        auto start = high_resolution_clock::now();
        for (int iteration = 0; iteration < iterationsNumber && !precisionReached(); iteration++) {
            if (xPrev == nullptr) {
                xPrev = new double[m];
            }
            for (int i = 0; i < m; i++) {
                xPrev[i] = x[i];
            }

            for (int i = startPosition; i < startPosition + length; i++) {
                double sum = 0;
                for (int j = 0; j < m; j++) {
                    if (i != j) {
                        sum += A[(i - startPosition) * m + j] * xPrev[j];
                    }
                }
                x[i] = (b[i - startPosition] - sum) / A[(i - startPosition) * m + i];
            }

            shareValuesIteration();
        }
        if (currentProcess == PRIMARY_PROCESS) {
            auto finish = high_resolution_clock::now();
            cout << "For m = " << m << " time is " << duration<double>(finish - start).count() << endl;
        }
        printResult();
    }

    bool precisionReached() {
        if (xPrev == nullptr) {
            return false;
        }
        for (int i = 0; i < m; i++) {
            if (abs(x[i] - xPrev[i]) > precision) {
                return false;
            }
        }
        return true;
    }

    void printResult() {
        if (currentProcess == PRIMARY_PROCESS) {
            ofstream stream(OUTPUT_FILE_NAME_2);
            stream << m << endl;
            for (auto i = 0; i < m; i++) {
                stream << x[i] << endl;
            }
            stream.close();
        }
    }

private:
    void readInput(char* coefficientsFile, char* xApproxFile) {
        ifstream coefStream(coefficientsFile);
        int n;
        coefStream >> m >> n;
        if (n != m + 1) {
            throw runtime_error("First file don't follow n = m + 1 rule!");
        }
        A = new double[m * m];
        b = new double[m];
        for (auto i = 0; i < m; i++) {
            for (auto j = 0; j < m; j++) {
                coefStream >> A[i * m + j];
            }
            coefStream >> b[i];
        }
        coefStream.close();

        ifstream xApproxStream(xApproxFile);
        xApproxStream >> n;
        if (n != m) {
            throw runtime_error("Second file 'm' value don't match the first one!");
        }
        x = new double[m];
        for (auto i = 0; i < m; i++) {
            xApproxStream >> x[i];
        }
        xApproxStream.close();
    }

    void generateInput(int m) {
        this->m = m;
        A = new double[m * m];
        b = new double[m];
        for (auto i = 0; i < m; i++) {
            double sum = 0;
            for (auto j = 0; j < m; j++) {
                A[i * m + j] = -10.0 + rand() % 20;
                if (i != j) sum += abs(A[i * m + j]);
            }
            b[i] = -1000.0 + rand() % 2000;
            // convergence condition
            if (A[i * m + i] < sum) A[i * m + i] = sum + 1.0;
        }
        x = new double[m];
        for (auto i = 0; i < m; i++) {
            x[i] = -10.0 + rand() % 20;
        }
    }

    void shareValuesIteration() {
        MPI_Status status;
        const int partLength = m / totalProcesses;
        for (int process = 0; process < totalProcesses; process++) {
            const int startPosition = process * partLength;
            const int length = (process < totalProcesses - 1) ? partLength : m - startPosition;
            MPI_Bcast(&x[startPosition], length, MPI_DOUBLE, process, MPI_COMM_WORLD);
        }
    }
};

int main22(int argc, char* argv[]) {
    MPI_Init(nullptr, nullptr);
    if (argc == 4) {
        // read files mode
        LinearEquatSolveProcess solver(argv[1], argv[2], stod(argv[3]));
        solver.computeValues(1000);
    }
    else {
        // random generated values mode
        LinearEquatSolveProcess solver1(300, 0.000001),
            solver2(800, 0.000001), solver3(2000, 0.000001);
        solver1.computeValues(1000);
        solver2.computeValues(1000);
        solver3.computeValues(1000);
    }
    MPI_Finalize();
    return 0;
}