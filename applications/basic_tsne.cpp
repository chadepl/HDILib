//
// Created by Nicolas Chaves on 2/3/20.
//

#include <iostream>
#include <fstream>
#include <hdi/utils/scoped_timers.h>
#include <hdi/utils/cout_log.h>
#include <hdi/utils/log_helper_functions.h>
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#include "hdi/dimensionality_reduction/sparse_tsne_user_def_probabilities.h"

#include "hdi/utils/Eigen/Dense"

// Global variables for matricial version of tSNE (mostly Eigen)


template<typename scalar_type>
int saveVectorSnapshot(const std::vector<scalar_type>& vector_to_save, int iteration){
    std::string path("../../../data/snapshots/ss_iter" + std::to_string(iteration) + ".csv");
    std::ofstream output_file (path);
    std::ostream_iterator<scalar_type> output_iterator(output_file, "\n");
    std::copy(vector_to_save.begin(), vector_to_save.end(), output_iterator);
}

template<typename scalar_type>
void writeEmbeddingToCSVfile(std::string name, const int& num_data_points, const int& num_target_dimensions, std::vector<scalar_type>& embedding_data)
{
    std::cout << embedding_data.size() << std::endl;
    Eigen::MatrixXf out(num_data_points, num_target_dimensions);

    for (int i = 0; i < num_data_points; ++i) {
        for (int j = 0; j < num_target_dimensions; ++j) {
            out(i,j) = embedding_data[i*num_target_dimensions + j];
        }
    }

    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(name.c_str());
    file << out.format(CSVFormat);
    file.close();
}

// from: https://stackoverflow.com/questions/18400596/how-can-a-eigen-matrix-be-written-to-file-in-csv-format
void writeToCSVfile(std::string name, Eigen::MatrixXf matrix)
{
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    file.close();
}

// from: https://gist.github.com/infusion/43bd2aa421790d5b4582
template<typename scalar_type>
void readFromCSVfile(std::string name, std::vector<scalar_type>& data, int num_dimensions)
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
                    data[row*num_dimensions + col++] = atof(start);
                    start = ptr + i + 1;
                }
            }
            data[row*num_dimensions + col++] = atof(start);

            row++;
        }
        in.close();
    }
}


int main(){
    std::cout << "INITIALIZING: t-SNE Acceleration Lab" << std::endl;

    const auto num_data_points = 150; // This is from the input file
    const auto num_dimensions = 4; // This is from the input file

    bool verbose                = true;
    int iterations              = 1;
    int exaggeration_iter       = 250;
    int perplexity              = 30;
    double theta                = 0;
    int num_target_dimensions   = 2;

    ////////////////////////////////////
    ////////////////////////////////////

    float data_loading_time = 0;
    float similarities_comp_time = 0;
    float gradient_desc_comp_time = 0;
    float data_saving_time = 0;

    ////////////////////////////////////
    ////////////////////////////////////

    hdi::utils::CoutLog log;

    typedef float scalar_type;

    std::vector<scalar_type> data;
    data.resize(num_data_points * num_dimensions);

    // Load binary data
//    {
//        hdi::utils::secureLog(&log, "1) Loading data ... ");
//        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_loading_time);
//        std::ifstream input_file (std::string("../../../data/MNIST_1000.bin"), std::ios::in|std::ios::binary|std::ios::ate);
//        std::cout << "Input size: " << input_file.tellg() << std::endl;
//        if(int(input_file.tellg()) != int(sizeof(scalar_type) * num_dimensions * num_data_points)){
//            std::cout << "Input file size doesn't agree with input parameters!" << std::endl;
//            return 1;
//        }
//        input_file.seekg (0, std::ios::beg);
//        input_file.read (reinterpret_cast<char*>(data.data()), sizeof(scalar_type) * num_dimensions * num_data_points);
//        input_file.close();
//    }

    {
        readFromCSVfile(std::string("../../../data/iris.csv"), data, num_dimensions);
        for (int r = 0; r < num_data_points; ++r) {
            for (int c = 0; c < num_dimensions; ++c) {
                std::cout << data[r * num_dimensions + c] << ", ";
            }
        }
    }

    ////////////////////////////////////
    ////////////////////////////////////

    hdi::dr::SparseTSNEUserDefProbabilities<scalar_type> tSNE;
    hdi::dr::TsneParameters tSNE_params;
    hdi::data::Embedding<scalar_type> embedding;

    hdi::dr::HDJointProbabilityGenerator<scalar_type> prob_generator;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::sparse_scalar_matrix_type distributions;
    hdi::dr::HDJointProbabilityGenerator<scalar_type>::Parameters prob_gen_params;
    
    {
        hdi::utils::secureLog(&log, "2) Computing high dimensional similarities ... ");
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(similarities_comp_time);
        prob_gen_params._perplexity = perplexity;
        prob_generator.computeProbabilityDistributions(data.data(), num_dimensions, num_data_points, distributions, prob_gen_params);

        Eigen::MatrixXf unnormalizedDist = Eigen::MatrixXf::Zero(num_data_points, num_data_points);

        for (int j = 0; j < num_data_points; ++j) {
            for (auto elem: distributions[j]) {
                unnormalizedDist(j, elem.first) = elem.second;
            }
        }
        writeToCSVfile(std::string("../../../data/debugging_files/unormalizedDistOriginal.csv"), unnormalizedDist);
    }

    ////////////////////////////////////
    ////////////////////////////////////

    {
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(gradient_desc_comp_time);
        tSNE_params._embedding_dimensionality = num_target_dimensions;
        tSNE_params._mom_switching_iter = exaggeration_iter;
        tSNE_params._remove_exaggeration_iter = exaggeration_iter;
        tSNE_params._seed = 1;
        tSNE.initialize(distributions, &embedding, tSNE_params);
        tSNE.setTheta(theta);


        hdi::utils::secureLog(&log, "3) Computing gradient descent ... ");
        for (int i = 0; i < iterations; ++i) {
            tSNE.doAnIteration();

            Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(num_data_points, num_data_points);
            for (int r = 0; r < num_data_points; ++r) {
                for (int c = 0; c < num_data_points; ++c) {
                    Q(r,c) = tSNE.getDistributionQ()[r*num_target_dimensions + c];

                }
            }
            writeToCSVfile(std::string("../../../data/outputs/normalQ.csv"), Q);

            Eigen::MatrixXf G = Eigen::MatrixXf::Zero(num_data_points, num_target_dimensions);
            for (int r = 0; r < num_data_points; ++r) {
                for (int c = 0; c < num_target_dimensions; ++c) {
                    G(r,c) = tSNE.getGradient()[r*num_target_dimensions + c];

                }
            }
            writeToCSVfile(std::string("../../../data/outputs/normalG.csv"), G);

            hdi::utils::secureLogValue(&log, "Iter", i, verbose);

        }
        hdi::utils::secureLog(&log, "- done!");
    }

    writeEmbeddingToCSVfile(std::string("../../../data/output_original.csv"), num_data_points, num_target_dimensions, embedding.getContainer());

    {
        hdi::utils::ScopedTimer<float, hdi::utils::Seconds> timer(data_saving_time);
        // Output Binary
        {
            std::ofstream output_file (std::string("../../../data/output.bin"), std::ios::out|std::ios::binary);
            output_file.write(reinterpret_cast<const char*>(embedding.getContainer().data()),sizeof(scalar_type)*embedding.getContainer().size());
        }
        // Output CSV
        {
            std::ofstream output_file (std::string("../../../data/output.csv"));
            std::ostream_iterator<scalar_type> output_iterator(output_file, "\n");
            std::copy(embedding.getContainer().begin(), embedding.getContainer().end(), output_iterator);
        }
    }

    std::cout << "Finished all steps" << std::endl;
    hdi::utils::secureLogValue(&log, "Data loading time (secs)", data_loading_time);
    hdi::utils::secureLogValue(&log, "Similarities computation time (secs)", similarities_comp_time);
    hdi::utils::secureLogValue(&log, "Gradient descent time (secs)", gradient_desc_comp_time);
    hdi::utils::secureLogValue(&log, "Data saving time (secs)", data_saving_time);

    return 0;
}
