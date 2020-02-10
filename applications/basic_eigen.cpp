//
// Created by Nicolas Chaves on 2/4/20.
//

#include <iostream>
#include "hdi/utils/Eigen/Dense"


void coefwiseSigns(Eigen::MatrixXf& input, Eigen::MatrixXf& output){
    Eigen::MatrixXf t1 = ((input.array() < 0).cast<float>() * -1);
    Eigen::MatrixXf t2 = (input.array() > 0).cast<float>();
    output = t1 + t2;
}

int main(){

    std::cout << "Hello world" << std::endl;

    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    Eigen::MatrixXf temp =  ((Eigen::MatrixXf::Random(4, 2).array() + 0) * 50).matrix();
    temp(0,1) = 0;
    Eigen::MatrixXf signedTemp(4,2);

//    std::cout << (temp.array() > 10.f) << std::endl;
    std::cout << temp << std::endl;
    coefwiseSigns(temp, signedTemp);
    std::cout << signedTemp << std::endl;

    // A gradient needs the following matrices:
    // - Weights (W) [n x n]
    // - High-dimensional Laplacian (L_P) [n x n]
    // - Low-dimensional Laplacian (L_Q) [n x n]

    // On top of this we have:
    // - Embedding (Y) at current iteration [n x 2]

    // Gradient computation is:
    // - 4 * W o (L_P - L_Q) * Y^T
    // [n x n] o ([n x n] - [n x n]) * [2 x n] CHECK

    auto num_data_points = 10;
    auto num_target_dimensions = 2;
    Eigen::MatrixXf LP(num_data_points, num_data_points);
    Eigen::MatrixXf LQ(num_data_points, num_data_points);
    Eigen::MatrixXf W(num_data_points, num_data_points);
    Eigen::MatrixXf Y(num_data_points, num_target_dimensions);
    Eigen::MatrixXf Gradient;

    for (int i = 0; i < num_data_points; ++i) {
        for (int j = 0; j < num_data_points; ++j) {
            LP(i, j) = 2;
            LQ(i, j) = 1;
            if(i == j)
                W(i, j) = 1;
            else
                W(i, j) = 0;
        }
        for (int k = 0; k < num_target_dimensions; ++k) {
            Y(i, k) = 1;
        }
    }

    {
        Eigen::MatrixXf L = LP - LQ;
        Gradient = (W.cwiseProduct(L) * 4) * Y;
        std::cout << Gradient << std::endl;
        std::cout << Gradient.rows() << std::endl;
        std::cout << Gradient.cols() << std::endl;
    }



    std::cout << "Size of P: " << LP.size()  << std::endl;
    std::cout << "Size of Q: " << LQ.size() << std::endl;
    std::cout << "Size of W: " << W.size() << std::endl;
    std::cout << "Size of Gradient: " << Gradient.size() << std::endl;

    return 1;
}

