#pragma once
#pragma warning(disable : 4996)
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iterator>
#include <algorithm>
#include <fstream>

#include <wchar.h>
#include <time.h>

#define x2(x) (x * x)
#define x(x) (x * x2(x))
#define _MAX_PATH 256

enum NORMALIZATION{NONE, MINMAX, ZSCORE, SIGMOIDAL, ENERGY};
wchar_t normalizations_type[][20] = {L"minmax", L"zscore", L"sigmoidal", L"energy"};

int normalization = 0;
int validation_type = 0;
int vector_length = 0; // size of first vector read from read_class() trn, vld or tst set

class network *ann =0;

typedef struct _entry {
  float* vec;      //vector
  int size;        // vector size
  wchar_t fname[_MAX_PATH];   // vector file name
  int cls;         // vector class
} ENTRY, *PENTRY;


typedef struct _rec {
  std::vector<PENTRY> entries;   //rec entries
  std::vector<std::vector<int> > indices;   // 2d array of classes indices
  std::vector<int>clsnum;        // classes numbers of induces columns
}REC, *PREC;

