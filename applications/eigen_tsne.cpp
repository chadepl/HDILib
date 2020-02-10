//
// Created by Nicolas Chaves on 2/5/20.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include <hdi/utils/timing_utils.h>
#include <hdi/utils/scoped_timers.h>
#include <hdi/utils/cout_log.h>
#include <hdi/utils/log_helper_functions.h>

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"

#include "hdi/utils/Eigen/Dense"

/////////// UTILS /////////////

// from: https://stackoverflow.com/questions/18400596/how-can-a-eigen-matrix-be-written-to-file-in-csv-format
void writeToCSVfile(std::string name, Eigen::MatrixXf matrix)
{
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    file.close();
}

// from: https://gist.github.com/infusion/43bd2aa421790d5b4582
void readFromCSVfile(std::string name, Eigen::MatrixXf& mat)
{
    std::ifstream in(name);
    std::string line;

    int row = 0;
    int col = 0;

    if(in.is_open()) {
        while(std::getline(in, line)){
            char *ptr = (char *) line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;
            for (int i = 0; i < len; i++) {

                if (ptr[i] == ',') {
                    mat(row, col++) = atof(start);
                    start = ptr + i + 1;
                }
            }
            mat(row, col) = atof(start);

            row++;
        }
        in.close();
    }
}

//////////////////////////////

template<typename scalar_type>
int loadX(const std::string& path, std::vector<scalar_type>& data, const int& num_data_points, const int& num_dimensions){
    std::ifstream input_file (path, std::ios::in|std::ios::binary|std::ios::ate);
    std::cout << "Input size: " << input_file.tellg() << std::endl;
    if(int(input_file.tellg()) != int(sizeof(scalar_type) * num_dimensions * num_data_points)){
        std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
        return -1;
    }
    input_file.seekg (0, std::ios::beg);
    input_file.read (reinterpret_cast<char*>(data.data()), sizeof(scalar_type) * num_dimensions * num_data_points);
    input_file.close();
    return 1;
}


// TODO: maybe matrixXf is too specific
template<typename scalar_type>
void computeHighDimTerms(int perplexity, std::vector<scalar_type>& data, int& num_data_points, int& num_dimensions, Eigen::MatrixXf& P, Eigen::MatrixXf& DP){
    typename hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_generator;
    typename hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distribution;
    typename hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_params;
    prob_gen_params._perplexity = perplexity;
    prob_generator.computeProbabilityDistributions(data.data(), num_dimensions, num_data_points, distribution, prob_gen_params);

    for(int j = 0; j < num_data_points; ++j){
        for(auto& elem: distribution[j]){
            scalar_type v0 = elem.second;
            auto iter = distribution[elem.first].find(j);
            scalar_type v1 = 0.;
            if(iter != distribution[elem.first].end())
                v1 = iter->second;

            P(j, elem.first) = static_cast<scalar_type>((v0+v1)*0.5);
            P(elem.first, j) = static_cast<scalar_type>((v0+v1)*0.5);
        }
    }

    auto diagonalPDegrees = P.rowwise().sum();
    for (int rc = 0; rc < num_data_points; ++rc) {
        DP(rc,rc) = diagonalPDegrees(rc);
    }
}

// Copied from atSNE implementation
void initializeEmbedding(Eigen::MatrixXf& Y, const int& num_data_points, const int& num_target_dimensions, const double multiplier){
    // Y = Eigen::MatrixXf::Random(num_data_points, num_target_dimensions); // These are being initialized between -1 and 1

    int seed = 1; // hardcoded
    if(seed < 0){
        std::srand(static_cast<unsigned int>(time(NULL)));
    }
    else{
        std::srand(seed);
    }

    for (int i = 0; i < num_data_points; ++i) {
        double x(0.);
        double y(0.);
        double radius(0.);
        do {
            x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
            y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
            radius = (x * x) + (y * y);
        } while((radius >= 1.0) || (radius == 0.0));

        radius = sqrt(-2 * log(radius) / radius);
        x *= radius * multiplier;
        y *= radius * multiplier;
        Y(i, 0) = x;
        Y(i, 1) = y;
    }
}

void computeLowDimTerms(const int& num_data_points, const int& num_target_dimensions, float& normalization_factor, const Eigen::MatrixXf& Y, Eigen::MatrixXf& Q, Eigen::MatrixXf& DQ, Eigen::MatrixXf& W){

    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = i + 1; j < Y.rows(); ++j) {
            auto temp = Y.row(i) - Y.row(j);
            auto euclidean_dist_sq = 1./(1. + temp.dot(temp));
            W(i, j) = euclidean_dist_sq;
            W(j, i) = euclidean_dist_sq;
            normalization_factor += euclidean_dist_sq;
        }
    }

    normalization_factor = normalization_factor * 2; // TODO: Check

    Q = W * (1./normalization_factor); // TODO: Check

    auto diagonalQDegrees = Q.rowwise().sum();
    for (int rc = 0; rc < num_data_points; ++rc) {
        DQ(rc,rc) = diagonalQDegrees(rc);
    }

}

// Copied from atSNE implementation
void computeExaggeration(int iteration, double exaggeration_baseline, hdi::dr::TsneParameters& params){
    double exaggeration = exaggeration_baseline;
    if(iteration <= params._remove_exaggeration_iter){
        exaggeration = params._exaggeration_factor;
    }else if(iteration <= (params._remove_exaggeration_iter + params._exponential_decay_iter)){
        //double decay = std::exp(-scalar_type(_iteration-_params._remove_exaggeration_iter)/30.);
        double decay = 1. - double(iteration-params._remove_exaggeration_iter)/params._exponential_decay_iter;
        exaggeration = exaggeration_baseline + (params._exaggeration_factor-exaggeration_baseline)*decay;
        //utils::secureLogValue(_logger,"Exaggeration decay...",exaggeration);
    }

    params._exaggeration_factor = exaggeration;
}


void computeGradient(unsigned int iteration, hdi::dr::TsneParameters& params, Eigen::MatrixXf& G, const Eigen::MatrixXf& Y, const Eigen::MatrixXf& LP, const Eigen::MatrixXf& LQ, const Eigen::MatrixXf& W){
    // ([n x n] o [n x n]) x [n x 2] Is this operation alright?
    Eigen::MatrixXf L = W.cwiseProduct(LP - LQ);
    writeToCSVfile(std::string("../../../data/temp.csv"), L);
    G = 4 * (L * Y); // TODO: not sure about exaggeration here
}


//temp
template <typename T>
T sign(T x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Copied from atSNE implementation
// Q: how is the gain initialized?
void updateY(unsigned int iteration, hdi::dr::TsneParameters& params, Eigen::MatrixXf& Y, Eigen::MatrixXf& G, Eigen::MatrixXf& prev_G, Eigen::MatrixXf& gain, float mult=1){

    for(int r = 0; r < G.rows(); ++r){
        for(int c = 0; c < G.cols(); ++c) {
            gain(r,c) =  (sign(G(r,c)) != sign(prev_G(r,c)) ? (gain(r,c) + .2): (gain(r,c) * .8));
            if (gain(r,c) < params._minimum_gain) {
                gain(r,c) = params._minimum_gain;
            }
            G(r,c) = (G(r,c) > 0 ? 1 : -1) * std::abs(G(r,c) * params._eta * gain(r,c)) /(params._eta * gain(r,c));

            prev_G(r,c) = ((iteration < params._mom_switching_iter) ? params._momentum : params._final_momentum) *
                    prev_G(r,c) - params._eta * gain(r,c) * G(r,c);
            Y(r,c) += G(r,c) * mult;
        }
    }

}

void performGradientDescentStep(unsigned int iteration, hdi::dr::TsneParameters& params, Eigen::MatrixXf& G, Eigen::MatrixXf& prev_G, Eigen::MatrixXf& Y, const Eigen::MatrixXf& P, const Eigen::MatrixXf& DP, Eigen::MatrixXf& LP, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& DQ, Eigen::MatrixXf& LQ, const Eigen::MatrixXf& W, Eigen::MatrixXf& gain){

    // Define Laplacians
    LP = DP - (P * params._exaggeration_factor); // Might change because of a different exaggeration
    LQ = DQ - Q;

    computeGradient(iteration, params, G, Y, LP, LQ, W);
    updateY(iteration, params, Y, G, prev_G, gain);

    //TODO: MAGIC NUMBER
//    if(params._exaggeration_factor) > 1.2){
//        _embedding->scaleIfSmallerThan(0.1);
//    }else{
//        _embedding->zeroCentered();
//    }
}


std::string getFilePath(std::string name){
    return std::string("../../../data/") + name;
}


int main(){

    typedef float scalar_type;

    std::string path(getFilePath("MNIST_1000.bin"));

    ////////////////////////////////////
    ////////////////////////////////////

    hdi::utils::CoutLog log;

    float data_loading_time = 0;
    float similarities_comp_time = 0;
    float gradient_desc_comp_time = 0;
    float data_saving_time = 0;

    ////////////////////////////////////
    ////////////////////////////////////

    int num_iterations = 1;
    int num_data_points = 150;//1000;
    int num_dimensions = 4;//784;
    int num_target_dimensions = 2;

    hdi::dr::TsneParameters tSNEparams;
    int perplexity = 30;
    float normalization_factor = 0.;
    const double exaggeration_baseline = 1.;

    Eigen::MatrixXf data_csv(num_data_points,num_dimensions);
    std::vector<scalar_type> data;
    data.resize(num_data_points * num_dimensions);

    //loadX(path, data, num_data_points, num_dimensions);
    readFromCSVfile(getFilePath(std::string("iris.csv")), data_csv);
    for (int r = 0; r < num_data_points; ++r) {
        for (int c = 0; c < num_dimensions; ++c) {
            data[r * num_dimensions + c] = data_csv(r,c);
        }
    }

    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf DP = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf LP = Eigen::MatrixXf::Zero(num_data_points,num_data_points);

    Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(num_data_points, num_target_dimensions);
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(num_data_points, num_data_points);
    Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(num_data_points, num_data_points);
    Eigen::MatrixXf DQ = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf LQ(num_data_points, num_data_points);

    Eigen::MatrixXf gain = Eigen::MatrixXf::Ones(Y.rows(), Y.cols());
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());
    Eigen::MatrixXf prev_G = Eigen::MatrixXf::Zero(Y.rows(), Y.cols());


    computeHighDimTerms(perplexity, data, num_data_points, num_dimensions, P, DP);
    initializeEmbedding(Y, num_data_points, num_target_dimensions, tSNEparams._rngRange);


    // Scoped block for the descent procedure
    {
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
        for (int i = 0; i < num_iterations; ++i) {

            std::cout << "Iteration: " << i << std::endl;

            normalization_factor = 0.;
            computeLowDimTerms(num_data_points, num_target_dimensions, normalization_factor, Y, Q, DQ, W);
            computeExaggeration(i, exaggeration_baseline, tSNEparams);

            //writeToCSVfile(getFilePath("outputs/eigenY.csv"), Y);
            writeToCSVfile(getFilePath("outputs/eigenQ.csv"), Q);
            writeToCSVfile(getFilePath("outputs/eigenW.csv"), W);

            performGradientDescentStep(i, tSNEparams, G, prev_G, Y, P, DP, LP, Q, DQ, LQ, W, gain);

            writeToCSVfile(getFilePath("outputs/eigenG.csv"), G);

        }
    }



    writeToCSVfile(std::string("../../../data/out_embedding.csv"), Y);

    hdi::utils::secureLogValue(&log, "Gradient descent time (secs)", gradient_desc_comp_time);

    return 1;

}
