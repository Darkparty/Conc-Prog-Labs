#include <iostream>
#include <fstream>
#include <mpi.h>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int PRIMARY_PROCESS = 0;
const char* OUTPUT_FILE_NAME_3 = "QuickSort.output";
const int BUFFER_ARRAY_SIZE = 10000000;

struct ArrayInt {
	int size;
	int* values;
};

ArrayInt createArrayInt(const int* values, int size) {
	ArrayInt result;
	result.size = size;
	result.values = new int[size];
	for (int i = 0; i < size; i++) {
		result.values[i] = values[i];
	}
	return result;
}

ArrayInt mergeArraysInt(ArrayInt first, ArrayInt second) {
	ArrayInt result;
	result.size = first.size + second.size;
	result.values = new int[result.size];
	for (int i = 0; i < first.size; i++) {
		result.values[i] = first.values[i];
	}
	for (int i = first.size; i < result.size; i++) {
		result.values[i] = second.values[i - first.size];
	}
	return result;
}

struct ProcessGroup {
	int currentProcess;
	int totalProcesses;
	MPI_Comm communicator;
};

ProcessGroup createProcessGroup(MPI_Comm communicator) {
	ProcessGroup result;
	result.communicator = communicator;
	MPI_Comm_rank(communicator, &result.currentProcess);
	MPI_Comm_size(communicator, &result.totalProcesses);
	return result;
}

bool isInLeftHalf(ProcessGroup processGroup) {
	return processGroup.currentProcess < processGroup.totalProcesses / 2;
}

int getPartner(ProcessGroup processGroup) {
	if (isInLeftHalf(processGroup)) {
		return processGroup.currentProcess + processGroup.totalProcesses / 2;
	}
	return processGroup.currentProcess - processGroup.totalProcesses / 2;
}

int compareInts(const void* a, const void* b) {
	int int1 = *((int*)a);
	int int2 = *((int*)b);
	return int1 - int2;
}

class QuicksortProcess {
public:
	int currentPivot = 0;
	ArrayInt currentArray;
	ProcessGroup globalGroup;
	ProcessGroup currentGroup;

	QuicksortProcess(string arrayFile) {
		globalGroup = createProcessGroup(MPI_COMM_WORLD);
		currentGroup = globalGroup;
		if (globalGroup.currentProcess == PRIMARY_PROCESS) {
			readInput(arrayFile);
		}
		initialize();
	}

	QuicksortProcess(int size) {
		globalGroup = createProcessGroup(MPI_COMM_WORLD);
		currentGroup = globalGroup;
		if (globalGroup.currentProcess == PRIMARY_PROCESS) {
			generateInput(size);
		}
		initialize();
	}

	void initialize() {
		if (globalGroup.currentProcess == PRIMARY_PROCESS) {
			MPI_Request request;
			for (int process = 0; process < globalGroup.totalProcesses; process++) {
				int partSize = currentArray.size / globalGroup.totalProcesses;
				int begin = process * partSize;
				if (process == globalGroup.totalProcesses - 1) {
					partSize = currentArray.size - begin;
				}
				auto part = createArrayInt(&currentArray.values[begin], partSize);
				MPI_Isend(part.values, part.size, MPI_INT, process, 0, MPI_COMM_WORLD, &request);
			}
		}

		MPI_Status status;
		int* buffer = new int[BUFFER_ARRAY_SIZE];
		MPI_Recv(buffer, BUFFER_ARRAY_SIZE, MPI_INT, PRIMARY_PROCESS, 0, MPI_COMM_WORLD, &status);

		int size;
		MPI_Get_count(&status, MPI_INT, &size);
		currentArray = createArrayInt(buffer, size);
	}

	void choosePivot() {
		if (currentGroup.currentProcess == PRIMARY_PROCESS) {
			// using median of 7 or less values
			int values[7];
			for (int i = 0; i < size(values); i++) {
				int index = 0 + currentArray.size / size(values) * i;
				if (index >= currentArray.size) {
					index = currentArray.size - 1;
				}
				values[i] = currentArray.values[index];
			}
			qsort(values, size(values), sizeof(int), compareInts);
			currentPivot = values[size(values) / 2];
		}
		MPI_Bcast(&currentPivot, 1, MPI_INT, PRIMARY_PROCESS, currentGroup.communicator);
	}

	void exchangeValuesWithPartner() {
		int partner = getPartner(currentGroup);

		ArrayInt low, high;
		int i = -1, j = currentArray.size;
		while (true) {
			do {
				i++;
			} while (currentArray.values[i] < currentPivot);
			do {
				j--;
			} while (currentArray.values[j] > currentPivot);
			if (i >= j) break;
			swap(currentArray.values[i], currentArray.values[j]);
		}
		j++;
		low = createArrayInt(currentArray.values, j);
		high = createArrayInt(&currentArray.values[j], currentArray.size - j);

		MPI_Request request;
		if (isInLeftHalf(currentGroup)) {
			MPI_Isend(high.values, high.size, MPI_INT, partner, 0, currentGroup.communicator, &request);
		}
		else {
			MPI_Isend(low.values, low.size, MPI_INT, partner, 0, currentGroup.communicator, &request);
		}

		MPI_Status status;
		int* buffer = new int[BUFFER_ARRAY_SIZE];
		MPI_Recv(buffer, BUFFER_ARRAY_SIZE, MPI_INT, partner, 0, currentGroup.communicator, &status);

		int size;
		MPI_Get_count(&status, MPI_INT, &size);
		if (isInLeftHalf(currentGroup)) {
			high = createArrayInt(buffer, size);
		}
		else {
			low = createArrayInt(buffer, size);
		}

		currentArray = mergeArraysInt(low, high);
	}

	void splitToSubgroups() {
		int newGroup = isInLeftHalf(currentGroup) ? 0 : 1;
		MPI_Comm newCommunicator;
		MPI_Comm_split(currentGroup.communicator, newGroup, 0, &newCommunicator);
		currentGroup = createProcessGroup(newCommunicator);
	}

	void sortAndPrintResult() {
		auto start = high_resolution_clock::now();
		while (true) {
			if (currentGroup.totalProcesses == 1) {
				qsort(currentArray.values, currentArray.size, sizeof(int), compareInts);
				break;
			}
			choosePivot();
			exchangeValuesWithPartner();
			splitToSubgroups();
		}

		if (globalGroup.currentProcess != PRIMARY_PROCESS) {
			MPI_Request req;
			MPI_Isend(currentArray.values, currentArray.size, MPI_INT, PRIMARY_PROCESS, 0, MPI_COMM_WORLD, &req);
		}
		else {
			ArrayInt result = mergeProcessesResults();
			auto finish = high_resolution_clock::now();
			cout << "For size = " << result.size << " and process count = " << globalGroup.totalProcesses << " time is " << duration<double>(finish - start).count() << endl;
			ofstream stream(OUTPUT_FILE_NAME_3);
			stream << result.size << endl;
			for (auto i = 0; i < result.size; i++) {
				stream << result.values[i] << " ";
			}
			stream.close();
		}
	}

	ArrayInt mergeProcessesResults() {
		ArrayInt result = currentArray;
		for (int process = 0; process < globalGroup.totalProcesses; process++) {
			if (process == PRIMARY_PROCESS) {
				continue;
			}

			MPI_Status status;
			int* buffer = new int[BUFFER_ARRAY_SIZE];
			MPI_Recv(buffer, BUFFER_ARRAY_SIZE, MPI_INT, process, 0, MPI_COMM_WORLD, &status);
			int size;
			MPI_Get_count(&status, MPI_INT, &size);

			ArrayInt array = createArrayInt(buffer, size);
			result = mergeArraysInt(result, array);
		}
		return result;
	}

private:
	void readInput(string arrayFile) {
		ifstream stream(arrayFile);
		int size;
		stream >> size;
		int* values = new int[size];
		for (auto i = 0; i < size; i++) {
			stream >> values[i];
		}
		stream.close();
		currentArray = createArrayInt(values, size);
	}

	void generateInput(int size) {
		int* values = new int[size];
		for (auto i = 0; i < size; i++) {
			values[i] = -1000000 + rand() * (INT_MAX / RAND_MAX) % 2000000;
		}
		currentArray = createArrayInt(values, size);
	}
};

int main(int argc, char* argv[]) {
	MPI_Init(nullptr, nullptr);
	if (argc == 2) {
		// read file mode
		QuicksortProcess quicksort(argv[1]);
		quicksort.sortAndPrintResult();
	} else {
		// random generated values mode
		QuicksortProcess quicksort(100000);
		quicksort.sortAndPrintResult();
	}

	MPI_Finalize();
	return 0;
}