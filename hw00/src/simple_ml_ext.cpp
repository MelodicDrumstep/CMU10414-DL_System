#define TEST

#ifdef TEST
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#endif 
#include <cmath>
#include <iostream>

#ifdef TEST
namespace py = pybind11;
#endif

//#define DEBUG

float * make_Z(const float * X, float * theta, size_t m, size_t n, size_t k);
float * make_I(const unsigned char * y, size_t m, size_t k);

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(int i = 0; i < (m + batch - 1) / batch; i++)
    {

        const float * start_x = X + i * batch * n;
        const unsigned char * start_y = y + i * batch;

        #ifdef DEBUG    
        std::cout << "i: " << i << std::endl;
        //std::cout << "start_x is : " << start_x << std::endl;
        //std::cout << "start_y is : " << start_y << std::endl;
        std::cout << "X is :" << std::endl;

        for(int j = 0; j < batch; j++)
        {
            for(int l = 0; l < n; l++)
            {
                std::cout << start_x[j * n + l] << " ";
            }
            std::cout << std::endl;
        }
        #endif 

        #ifdef DEBUG
        std::cout << "theta is :" << std::endl;

        for(int j = 0; j < n; j++)
        {
            for(int l = 0; l < k; l++)
            {
                std::cout << theta[j * k + l] << " ";
            }
            std::cout << std::endl;
        }
        #endif

        float * Z = make_Z(start_x, theta, batch, n, k);


        #ifdef DEBUG
        std::cout << "Z is :" << std::endl;

        for(int j = 0; j < batch; j++)
        {
            for(int l = 0; l < k; l++)
            {
                std::cout << Z[j * k + l] << " ";
            }
            std::cout << std::endl;
        }
        #endif

        float * I = make_I(start_y, batch, k);

        // gradient = (x.T @ (Z - I)) / batch 
        float * gradient = new float[n * k];
        for(int j = 0; j < n * k; j++)
        {
            gradient[j] = 0;
        }
        for(int j = 0; j < n; j++)
        {
            for(int i = 0; i < m; i++)
            {
                for(int l = 0; l < k; l++)
                {
                    gradient[j * k + l] += (start_x[j * m + i] * (Z[i * k + l] - I[i * k + l])) / batch;
                }
            }
        }

        for(int j = 0; j < n; j++)
        {
            for(int l = 0; l < k; l++)
            {
                theta[j * k + l] -= lr * gradient[j * k + l];
            }
        }

        delete [] gradient;
        delete [] I;
        delete [] Z;
    }
    /// END YOUR CODE
}

float * make_Z(const float * X, float * theta, size_t m, size_t n, size_t k)
{
    float * Z = new float[m * k];
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            Z[i * k + j] = 0;
        }
    }
    // Z = exp(X * theta)
    for(int i = 0; i < m; i++)
    {
        for(int l = 0; l < n; l++)
        {
            float x = X[i * n + l];
            for(int j = 0; j < k; j++)
            {
                Z[i * k + j] += x * theta[l * k + j];
            }
        }
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            Z[i * k + j] = exp(Z[i * k + j]);
        }
    }
    
    //normalization: 
    for(int i = 0; i < m; i++)
    {
        float sum = 0;
        for(int j = 0; j < k; j++)
        {
            sum += Z[i * k + j];
        }
        for(int j = 0; j < k; j++)
        {
            Z[i * k + j] /= sum;
        }
    }
    return Z;
}

float * make_I(const unsigned char * start_y, size_t m, size_t k)
{
    float * I = new float[m * k];
    for(int j = 0; j < m; j++)
    {
        for(int l = 0; l < k; l++)
        {
            I[j * k + l] = 0;
        }
        I[j * k + start_y[j]] = 1;
    }
    return I;
}

#ifdef TEST
/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
#endif

#ifdef DEBUG

#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>

// Helper function to compare arrays
template <typename T>
bool compare_arrays(const T* a, const T* b, size_t size, T epsilon = 1e-5) 
{
    return std::equal(a, a + size, b, [epsilon](T x, T y) { return std::fabs(x - y) < epsilon; });
}

// Unit test for make_Z function
void test_make_Z() {
    // Set up test data
    size_t m = 2; // number of examples
    size_t n = 2; // input dimensions
    size_t k = 2; // number of classes
    std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f}; // m * n
    std::vector<float> theta = {0.1f, 0.2f, 0.3f, 0.4f}; // n * k
    std::vector<float> expected_Z = { // Expected Z after softmax
        std::exp(1.0f * 0.1f + 2.0f * 0.3f), std::exp(1.0f * 0.2f + 2.0f * 0.4f),
        std::exp(3.0f * 0.1f + 4.0f * 0.3f), std::exp(3.0f * 0.2f + 4.0f * 0.4f)
    };

    // Normalize expected Z
    float sum_row1 = expected_Z[0] + expected_Z[1];
    float sum_row2 = expected_Z[2] + expected_Z[3];
    expected_Z[0] /= sum_row1;
    expected_Z[1] /= sum_row1;
    expected_Z[2] /= sum_row2;
    expected_Z[3] /= sum_row2;

    float* Z = make_Z(X.data(), theta.data(), m, n, k);

    std::cout << "Z is : " << std::endl;

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            std::cout << Z[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "expected Z is : " << std::endl;

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < k; j++)
        {
            std::cout << expected_Z[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Check if the computed Z matches the expected Z
    std::cout << "the result : " << compare_arrays(Z, expected_Z.data(), m * k) << std::endl;
    
    delete[] Z; // Clean up

    std::cout << "test_make_Z passed." << std::endl;
}

// Unit test for make_I function
void test_make_I() {
    // Set up test data
    size_t m = 2; // number of examples
    size_t k = 3; // number of classes
    std::vector<unsigned char> y = {0, 2}; // Class labels
    std::vector<float> expected_I = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // One-hot encoded labels

    float* I = make_I(y.data(), m, k);

    // Check if the computed I matches the expected I
    assert(compare_arrays(I, expected_I.data(), m * k));

    delete[] I; // Clean up

    std::cout << "test_make_I passed." << std::endl;
}

int main()
{
    size_t m = 2;
    size_t n = 2;
    size_t k = 2;
    float * X = new float[m * n];
    unsigned char * y = new unsigned char[m];
    float * theta = new float[n * k];
    float lr = 0.1;
    int batch = 1;
    for(int i = 0; i < m * n; i++)
    {
        X[i] = i;
    }
    for(int i = 0; i < m; i++)
    {
        y[i] = 0;
    }
    y[0] = 1;
    for(int i = 0; i < n * k; i++)
    {
        theta[i] = 1;
    }
    test_make_I();
    test_make_Z();
}


#endif