// C program to find Burrows Wheeler transform 
// of a given text 

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 






// Structure to store data of a rotation 
struct rotation { 
	int index; 
	char* suffix; 
}; 


// function for comparing two strings. This function 
// is passed as a parameter to _quickSort() when we 
// want to sort 
int cmpstr(void* v1, void* v2) 
{ 
	// casting v1 to char** and then assigning it to 
	// pointer to v1 as v1 is array of characters i.e 
	// strings. 
	char *a1 = *(char**)v1; 
	char *a2 = *(char**)v2; 
	return strcmp(a1, a2); 
} 

// function for comparing two strings 
int cmpnum(void* s1, void* s2) 
{ 
	// casting s1 to int* so it can be 
	// copied in variable a. 
	int *a = (int*)s1; 
	int *b = (int*)s2; 
	if ((*a) > (*b)) 
		return 1; 
	else if ((*a) < (*b)) 
		return -1; 
	else
		return 0; 
} 

/* you can also write compare function for floats, 
	chars, double similarly as integer. */
// function for swap two elements 
void swap(void* v1, void* v2, int size) 
{ 
	// buffer is array of characters which will 
	// store element byte by byte 
	char buffer[size]; 

	// memcpy will copy the contents from starting 
	// address of v1 to length of size in buffer 
	// byte by byte. 
	memcpy(buffer, v1, size); 
	memcpy(v1, v2, size); 
	memcpy(v2, buffer, size); 
} 

// v is an array of elements to sort. 
// size is the number of elements in array 
// left and right is start and end of array 
//(*comp)(void*, void*) is a pointer to a function 
// which accepts two void* as its parameter 
void _qsort(void* v, int size, int left, int right, 
					int (*comp)(void*, void*)) 
{ 
	void *vt, *v3; 
	int i, last, mid = (left + right) / 2; 
	if (left >= right) 
		return; 

	// casting void* to char* so that operations 
	// can be done. 
	void* vl = (char*)(v + (left * size)); 
	void* vr = (char*)(v + (mid * size)); 
	swap(vl, vr, size); 
	last = left; 
	for (i = left + 1; i <= right; i++) { 

		// vl and vt will have the starting address 
		// of the elements which will be passed to 
		// comp function. 
		vt = (char*)(v + (i * size)); 
		if ((*comp)(vl, vt) > 0) { 
			++last; 
			v3 = (char*)(v + (last * size)); 
			swap(vt, v3, size); 
		} 
	} 
	v3 = (char*)(v + (last * size)); 
	swap(vl, v3, size); 
	_qsort(v, size, left, last - 1, comp); 
	_qsort(v, size, last + 1, right, comp); 
} 




// Compares the rotations and 
// sorts the rotations alphabetically 
int cmpfunc(void* x, void* y) 
{ 
	struct rotation* rx = (struct rotation*)x; 
	struct rotation* ry = (struct rotation*)y; 
	return strcmp(rx->suffix, ry->suffix); 
} 

// Takes text to be transformed and its length as 
// arguments and returns the corresponding suffix array 
int* computeSuffixArray(char* input_text, int len_text) 
{ 
	// Array of structures to store rotations and 
	// their indexes 
	struct rotation suff[len_text];
	int i=0;

	// Structure is needed to maintain old indexes of 
	// rotations after sorting them 
	for (i = 0; i < len_text; i++) { 
		suff[i].index = i; 
		suff[i].suffix = (input_text + i); 
	} 

	// Sorts rotations using comparison 
	// function defined above 
	_qsort(suff,sizeof(struct rotation) ,0,len_text-1, cmpfunc); 

	// Stores the indexes of sorted rotations 
	int* suffix_arr 
		= (int*)malloc(len_text * sizeof(int)); 
	for (i = 0; i < len_text; i++) 
		suffix_arr[i] = suff[i].index; 

	// Returns the computed suffix array 
	return suffix_arr; 
} 

// Takes suffix array and its size 
// as arguments and returns the 
// Burrows - Wheeler Transform of given text 
char* findLastChar(char* input_text, 
				int* suffix_arr, int n) 
{ 
	// Iterates over the suffix array to find 
	// the last char of each cyclic rotation 
	char* bwt_arr = (char*)malloc(n * sizeof(char)); 
	int i; 
	for (i = 0; i < n; i++) { 
		// Computes the last char which is given by 
		// input_text[(suffix_arr[i] + n - 1) % n] 
		int j = suffix_arr[i] - 1; 
		if (j < 0) 
			j = j + n; 

		bwt_arr[i] = input_text[j]; 
	} 

	bwt_arr[i] = '\0'; 

	// Returns the computed Burrows - Wheeler Transform 
	return bwt_arr; 
} 

// Driver program to test functions above 
int main() 
{ 
	char input_text[] = "banana$"; 
	int len_text = strlen(input_text); 

	// Computes the suffix array of our text 
	int* suffix_arr 
		= computeSuffixArray(input_text, len_text); 

	// Adds to the output array the last char 
	// of each rotation 
	char* bwt_arr 
		= findLastChar(input_text, suffix_arr, len_text); 

	printf("Input text : %s\n", input_text); 
	printf("Burrows - Wheeler Transform : %s\n", 
		bwt_arr); 
	return 0; 
} 

