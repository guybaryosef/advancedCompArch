/**
 * AlgoParams.cpp - Header File for the
 * axlgorithm parameters class.
 */


#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>


#define DEFAULT_K   50
#define DEFAULT_B   250
#define DEFAULT_W   200
#define DEFAULT_L   60
#define DEFAULT_L_c 30
#define DEFAULT_L_t  4
#define DEFAULT_L_l 30
#define FILTER_SIZE 100

class AlgoParams
{

private:
    bool                valid;

    unsigned            B;
    unsigned            K;
    unsigned            W;
    unsigned            L;
    unsigned            L_c;
    unsigned            L_t;
    unsigned            L_l;

    std::string         input_name;
    std::string         output_name;

    std::vector<int>    input_vec;
    std::vector<int>    filter_t;
    std::vector<int>    filter_f;

public:
    AlgoParams(int argv, char **argc);

    void             loadInputVectorFromFile();
    void             createFilter();
    void             writeOutput(const std::vector<int> &output_vec);

    bool             isValid();
    unsigned         getB();
    unsigned         getK();
    unsigned         getW();
    unsigned         getL();
    unsigned         getL_c();
    unsigned         getL_t();
    unsigned         getL_l();
    std::vector<int> getInputVec();
    std::vector<int> getFilter_t();
    std::vector<int> getFilter_f();
};


///  INLINE FUNCTIONS ///
inline bool             AlgoParams::isValid()       { return valid;     }
inline unsigned         AlgoParams::getB()          { return B;         }
inline unsigned         AlgoParams::getK()          { return K;         }
inline unsigned         AlgoParams::getW()          { return W;         }
inline unsigned         AlgoParams::getL()          { return L;         }
inline unsigned         AlgoParams::getL_c()        { return L_c;       }
inline unsigned         AlgoParams::getL_t()        { return L_t;       }
inline unsigned         AlgoParams::getL_l()        { return L_l;       }
inline std::vector<int> AlgoParams::getInputVec()   { return input_vec; }
inline std::vector<int> AlgoParams::getFilter_t()   { return filter_t;  }
inline std::vector<int> AlgoParams::getFilter_f()   { return filter_f;  }