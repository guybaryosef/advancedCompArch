/**
 * AlgoParams.cpp - Header File for the
 * algorithm parameters class.
 */


#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>


#define DEFAULT_K       25
#define DEFAULT_N_BIN   10


class AlgoParams
{

private:
    bool                valid;
    unsigned            n_bins;
    unsigned            k;
    std::string         input_name;
    std::string         output_name;
    std::vector<int>    input_vec;
    std::vector<int>    output_vec;

public:
    AlgoParams(const int argv, char **argc);

    void             loadInputVectorFromFile(const std::string &file_name);

    bool             isValid();
    unsigned         getNBins();
    unsigned         getK();
    std::vector<int> getInputVec();
    int              getInputVecLen();

};


///  INLINE FUNCTIONS ///
inline bool             AlgoParams::isValid()      { return valid;  }
inline unsigned         AlgoParams::getNBins()      { return n_bins;    }
inline unsigned         AlgoParams::getK()          { return k;         }
inline std::vector<int> AlgoParams::getInputVec()   { return input_vec; }
inline int              AlgoParams::getInputVecLen(){ return input_vec.size(); }
