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

std::string getFilePath(std::string name){
    return std::string("../../../data/") + name;
}

//////////////////////////////

// Computes the cost function of the method
template<typename scalar_type>
double computeKullbackLeiblerDivergence(const Eigen::Matrix<scalar_type,Eigen::Dynamic,Eigen::Dynamic>& P,const Eigen::Matrix<scalar_type,Eigen::Dynamic,Eigen::Dynamic>& Q){
    double energy = 0.;
    for (int r = 0; r < P.rows(); ++r) {
        for (int c = 0; c < P.cols(); ++c) {
            if(r == c || P(r,c) == 0)
                continue;
            energy += (P(r,c) * std::log(P(r,c)) - P(r,c) * std::log(Q(r,c)));
        }
    }
    return energy;
}

template<typename scalar_type>
double computeKLDivP(const Eigen::Matrix<scalar_type,Eigen::Dynamic,Eigen::Dynamic>& P){
    double out = 0.;
    for (int r = 0; r < P.rows(); ++r) {
        for (int c = 0; c < P.cols(); ++c) {
            if(r == c || P(r,c) == 0)
                continue;
            out += P(r,c) * std::log(P(r,c));
        }
    }
    return out;
}

template<typename scalar_type>
double computeKLDivQ(const Eigen::Matrix<scalar_type,Eigen::Dynamic,Eigen::Dynamic>& P, const Eigen::Matrix<scalar_type,Eigen::Dynamic,Eigen::Dynamic>& Q){
    double out = 0.;
    for (int r = 0; r < P.rows(); ++r) {
        for (int c = 0; c < P.cols(); ++c) {
            if(r == c || P(r,c) == 0)
                continue;
            out -= P(r,c) * std::log(Q(r,c));
        }
    }
    return out;
}


// TODO: maybe matrixXf is too specific
template<typename scalar_type>
void computeHighDimTerms(int perplexity, std::vector<scalar_type>& data, int& num_data_points, int& num_dimensions, Eigen::MatrixXf& P){
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

    P = P/static_cast<scalar_type>(num_data_points); // Making sure everythings adds up to 1

}

// Copied from atSNE implementation
void initializeEmbedding(Eigen::MatrixXf& Y, const int& num_data_points, const int& num_target_dimensions, const double multiplier, int seed = -1){

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

void computeLowDimTerms(const Eigen::MatrixXf& Y, Eigen::MatrixXf& Q, Eigen::MatrixXf& N){

    float normalization_factor = 0.;

    for (int i = 0; i < Y.rows(); ++i) {
        for (int j = i + 1; j < Y.rows(); ++j) {
            auto v1 = Y.row(i);
            auto v2 = Y.row(j);
            auto temp = v1 - v2;
            auto euclidean_dist_sq = temp.squaredNorm();
            auto q_kernel = 1./(1. + euclidean_dist_sq);
            N(i, j) = q_kernel;
            N(j, i) = q_kernel;
            normalization_factor += q_kernel;
        }
    }

    normalization_factor = normalization_factor * 2.; // TODO: Check

    Q = N * (1./normalization_factor); // TODO: Check
}


Eigen::MatrixXf computeGradient(const Eigen::MatrixXf Y, const Eigen::MatrixXf N, const Eigen::MatrixXf P, const Eigen::MatrixXf Q){
    Eigen::MatrixXf W = N.cwiseProduct(P - Q);
    Eigen::MatrixXf D = W.rowwise().sum().asDiagonal();
    Eigen::MatrixXf L = D - W;
    return 4 * L * Y;
}

Eigen::MatrixXf computeHessian(const Eigen::MatrixXf Y, const Eigen::MatrixXf N, const Eigen::MatrixXf P, const Eigen::MatrixXf Q){
    // 4L @ I_d + 8L^{xx} - 16 lambda vec(XL^q) vec(XL^q)^T
    // lambda = 1 for tsne
    return Eigen::MatrixXf::Zero(10,10); // TODO
}

// returns a signs matrix of a matrix
// a position is 1 if its larger than 0, -1 if its smaller and 0 if its exactly 0
Eigen::MatrixXf sign(const Eigen::MatrixXf& mat){
    Eigen::MatrixXf larger = (mat.array() > 0).cast<float>();
    Eigen::MatrixXf smaller = (mat.array() < 0).cast<float>() * -1;
    return smaller + larger;
}

// Method 1 is the VDM approach (momentum + goodies): copied from matlab implementation
void doIterationM1(const unsigned int iteration, Eigen::MatrixXf& Y, Eigen::MatrixXf& G, Eigen::MatrixXf& increments, Eigen::MatrixXf& gains, const Eigen::MatrixXf& P, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& N){

    float early_exaggeration;
    unsigned int stop_lying_iteration = 100;
    unsigned int momentum_switching_iteration = 250;
    float momentum = 0.5;
    float final_momentum = 0.8;
    float epsilon = 500;

    if(iteration < stop_lying_iteration)
        early_exaggeration = 4.;
    else
        early_exaggeration = 1.;

    if(iteration > momentum_switching_iteration)
        momentum = final_momentum;

    //Eigen::MatrixXf W = N.cwiseProduct((early_exaggeration * P) - Q);
    //Eigen::MatrixXf D = W.rowwise().sum().asDiagonal();
    //Eigen::MatrixXf L = D - W;
    //G = 4 * L * Y;

    G = computeGradient(Y, N, early_exaggeration * P, Q);


    Eigen::MatrixXf signsG = sign(G);
    Eigen::MatrixXf signsIncrements = sign(increments);
    Eigen::MatrixXf s_diff = (signsG.array() != signsIncrements.array()).cast<float>().matrix();
    Eigen::MatrixXf s_eq = (signsG.array() == signsIncrements.array()).cast<float>().matrix();

    gains = ((gains.array() + .2).matrix()).cwiseProduct(s_diff) + (gains * .8).cwiseProduct(s_eq);
    for (int r = 0; r < gains.rows(); ++r) {
        for (int c = 0; c < gains.cols(); ++c) {
            if(gains(r,c) < 0.01)
                gains(r,c) = 0.01;
        }
    }
    increments = (momentum * increments) - (epsilon * (gains.cwiseProduct(G)));

    Y = Y + increments;

    if(iteration % 100 == 0)
        writeToCSVfile(std::string("../../../data/snapshots/") + std::to_string(iteration) + std::string(".csv"), Y);
}


float lineSearchBacktracking(float alpha_bar, float KLDivP, float KLDivQ, const Eigen::MatrixXf& Y, Eigen::MatrixXf& p, Eigen::MatrixXf& G, const Eigen::MatrixXf& P){
    float alpha = alpha_bar;
    float rho = 0.5;
    float c = 0.5;

    Eigen::MatrixXf Y_prime = Y + (alpha * p);
    Eigen::MatrixXf Q_prime = Eigen::MatrixXf::Zero(P.rows(),P.rows());
    Eigen::MatrixXf N_prime = Eigen::MatrixXf::Zero(P.rows(),P.rows());
    computeLowDimTerms(Y_prime, Q_prime, N_prime);

    float KLDivQ_prime = computeKLDivQ(P, Q_prime);

    Eigen::Map<Eigen::VectorXf> G_vec(G.data(), G.size());
    Eigen::Map<Eigen::VectorXf> p_vec(p.data(), p.size());

    // val func(x + alpha * p) > func(x) + c * alpha * G^T * p
    //auto leftSide = KLDivP + KLDivQ_prime;
    auto temp = (c * alpha * (G.transpose() * p));
    //auto rightSide = (KLDivP + KLDivQ) + (c * alpha * (G.transpose() * p));
//    while (leftSide > rightSide){
//        alpha = rho * alpha;
//        std::cout << "-- alpha: " << alpha << std::endl;
//    }

    return alpha;
}

float lineSearchWolfe(){


}

// Method 2: steepest descent (gradient without any fancy stuff)
void doIterationM2(const unsigned int iteration, const float KLDivP, float& KLDivQ, Eigen::MatrixXf& Y, Eigen::MatrixXf& G, const Eigen::MatrixXf& P, Eigen::MatrixXf& Q, Eigen::MatrixXf& N){

    G = computeGradient(Y, N, P, Q);
    Eigen::MatrixXf p = G * -1;

    float epsilon = lineSearchBacktracking(500, KLDivP, KLDivQ, Y, p, G, P);

    Y = Y + (epsilon * p);

    computeLowDimTerms(Y, Q, N);

    KLDivQ = computeKLDivQ(P, Q);

    std::cout << " - Energy: " << KLDivP + KLDivQ << std::endl;
}

// Method 3: steepest descent with  backtracking line search(gradient without any fancy stuff)
void doIterationM3(const unsigned int iteration, Eigen::MatrixXf& G, Eigen::MatrixXf& Y, const Eigen::MatrixXf& P, const Eigen::MatrixXf& Q, const Eigen::MatrixXf& N){

    float epsilon = 500; // TODO

    G = computeGradient(Y, N, P, Q);
    Y = Y - (epsilon * G);
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

    int num_iterations = 100;
    int num_data_points = 1000;//1000;
    int num_dimensions = 784;//784;
    int num_target_dimensions = 2;

    hdi::dr::TsneParameters tSNEparams;
    int perplexity = 30;
    const double exaggeration_baseline = 1.;

    Eigen::MatrixXf data_csv(num_data_points,num_dimensions);
    std::vector<scalar_type> data;
    data.resize(num_data_points * num_dimensions);

    loadX(path, data, num_data_points, num_dimensions);
//    readFromCSVfile(getFilePath(std::string("iris.csv")), data_csv);
//    for (int r = 0; r < num_data_points; ++r) {
//        for (int c = 0; c < num_dimensions; ++c) {
//            data[r * num_dimensions + c] = data_csv(r,c);
//        }
//    }

    Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(num_data_points,num_target_dimensions);
    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf N = Eigen::MatrixXf::Zero(num_data_points,num_data_points);
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(num_data_points,num_target_dimensions);

    computeHighDimTerms(perplexity, data, num_data_points, num_dimensions, P);
    initializeEmbedding(Y, num_data_points, num_target_dimensions, tSNEparams._rngRange, -1);

    float KLDivP = computeKLDivP(P);
    float KLDivQ;

    // Scoped block for the descent procedure
    {
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);

        Eigen::MatrixXf increments = Eigen::MatrixXf::Zero(Y.rows(), Y.cols()); // for M1
        Eigen::MatrixXf gains = Eigen::MatrixXf::Ones(Y.rows(), Y.cols()); // for M1

        computeLowDimTerms(Y, Q, N);
        KLDivQ = computeKLDivQ(P, Q);

        for (int i = 0; i < num_iterations; ++i) {

            std::cout << "Iteration: " << i << std::endl;

            doIterationM2(i, KLDivP, KLDivQ, Y, G, P, Q, N);


            //std::cout << increments.block(0,0,4,2) << std::endl;
            //std::cout << std::endl;
            //std::cout << gains.block(0,0,4,2) << std::endl;

            //doIterationM1(i, Y, G, increments, gains, P, Q, N);
        }
    }



    writeToCSVfile(std::string("../../../data/out_embedding.csv"), Y);

    hdi::utils::secureLogValue(&log, "Gradient descent time (secs)", gradient_desc_comp_time);

    return 1;

}


//    std::vector<float> data = {3.38516029296219, 3.47327288367667,	5.17956148619538,
//                        6.84008049170251,	7.02285918507085,	3.98068174167929,
//                        4.18107424342671,	5.55229595044112,	5.03759946511528,
//                        5.99213557479753,	6.82014240762686,	5.13712330183306,
//                        5.53381778799157,	5.34264448640063,	3.47894773805727,
//                        -5.01891845321843,	-5.57374469161162,	-5.87411517559351,
//                        -4.83675332222214,	-4.89460780775250,	-4.06840610827687,
//                        -5.72115067859456,	-5.60509605355657,	-4.39995793880156,
//                        -4.58939664096957,	-4.57820537729022,	-4.42863931193692,
//                        -6.21263109313471,	-5.36280894288715,	-4.31488980818312};
//
//    int n = 10;
//    int d = 3;
//    int td = 2;
//    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(10,10);
//    Eigen::MatrixXf N = Eigen::MatrixXf::Zero(10,10);
//    Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(10,10);
//    //Eigen::MatrixXf Y = Eigen::MatrixXf::Zero(10,2);
//    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(10,2);
//
//    for (int i = 0; i < n; ++i) {
//        std::cout << data[i*d] << " "  << data[i*d+1] << " " << data[i*d+2] << std::endl;
//    }
//
//    //initializeEmbedding(Y, n, td, 1, -1);
//    Eigen::MatrixXf Y = Eigen::MatrixXf(10,2);
//    Y << 0.0333e-3,   -0.1603e-3,
//    0.0141e-3,   -0.2362e-3,
//    0.1578e-3,   -0.0702e-3,
//    0.0090e-3,    0.1652e-3,
//    -0.0673e-3,    0.0235e-3,
//    0.0932e-3,   -0.0152e-3,
//    -0.0358e-3,   -0.0156e-3,
//    0.0202e-3,    0.1038e-3,
//    0.0876e-3,    0.0330e-3,
//    0.0808e-3,    0.0476e-3;
//    computeHighDimTerms(2, data, n, d, P);
//    computeLowDimTerms(Y, Q, N),
//
//    std::cout << Y << std::endl; // OK
//    std::cout << std::endl;
//    std::cout << P << std::endl; // OK
//    std::cout << std::endl;
//    std::cout << Q << std::endl; // OK
//
//    Eigen::MatrixXf W = N.cwiseProduct((4 * P) - Q);
//    Eigen::MatrixXf D = W.rowwise().sum().asDiagonal();
//    Eigen::MatrixXf L = D - W;
//    G = 4 * L * Y;
//
//    std::cout << std::endl;
//    std::cout << W << std::endl; // OK
//    std::cout << std::endl;
//    std::cout << D << std::endl; // OK
//    std::cout << std::endl;
//    std::cout << L << std::endl; // OK
//    std::cout << std::endl;
//    std::cout << G << std::endl; // OK
//
//    Eigen::MatrixXf increments = Eigen::MatrixXf::Zero(10, 2);
//    Eigen::MatrixXf gains = Eigen::MatrixXf::Ones(10, 2);
//
//    Eigen::MatrixXf signsG = sign(G);
//    Eigen::MatrixXf signsIncrements = sign(increments);
//    Eigen::MatrixXf s_diff = (signsG.array() != signsIncrements.array()).cast<float>().matrix();
//    Eigen::MatrixXf s_eq = (signsG.array() == signsIncrements.array()).cast<float>().matrix();
//
//    gains = ((gains.array() + .2).matrix()).cwiseProduct(s_diff) + (gains * .8).cwiseProduct(s_eq);
//    for (int r = 0; r < gains.rows(); ++r) {
//        for (int c = 0; c < gains.cols(); ++c) {
//            if(gains(r,c) < 0.01)
//                gains(r,c) = 0.01;
//        }
//    }
//    increments = (0.5 * increments) - (500 * (gains.cwiseProduct(G)));
//
//    std::cout << std::endl;
//    std::cout << increments << std::endl;
//    std::cout << std::endl;
//    std::cout << gains << std::endl;
