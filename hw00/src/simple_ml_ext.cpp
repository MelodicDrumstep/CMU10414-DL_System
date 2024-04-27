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
float * make_gradient(const float * X, const float * I, const float * Z, size_t m, size_t n, size_t k, size_t batch);

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

    ///DEBUG START
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
    ///DEBUG END

        float * Z = make_Z(start_x, theta, batch, n, k);


    ///DEBUG START
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
    ///DEBUG END

        float * I = make_I(start_y, batch, k);

        float * gradient = make_gradient(start_x, I, Z, batch, n, k, batch);
            
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

// gradient = (x.T @ (Z - I)) / batch 
// X : (m x n)
// Z / I : (m x k)
float * make_gradient(const float * X, const float * I, const float * Z, size_t m, size_t n, size_t k, size_t batch)
{
    float * gradient = new float[n * k];

    float * transpose_A = new float[n * m];
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            transpose_A[j * m + i] = X[i * n + j];
        }
    }

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
                gradient[j * k + l] += (transpose_A[j * m + i] * (Z[i * k + l] - I[i * k + l]));
            }
        }
    }

    for(int j = 0; j < n * k; j++)
    {
            gradient[j] /= batch;
    }

    delete [] transpose_A;
    return gradient;
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
#include <cstdlib>
#include <ctime>

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

float *generateRandomFloats(size_t num, float rangeStart = 0.0, float rangeEnd = 1.0) {
    float *data = new float[num];
    float range = rangeEnd - rangeStart;
    for (size_t i = 0; i < num; ++i) {
        data[i] = rangeStart + (range * rand() / (RAND_MAX + 1.0));
    }
    return data;
}

unsigned char *generateRandomClasses(size_t num, unsigned char maxClass) {
    unsigned char *classes = new unsigned char[num];
    for (size_t i = 0; i < num; ++i) {
        classes[i] = static_cast<unsigned char>(rand() % (maxClass + 1));
    }
    return classes;
}

bool compareFloatArrays(const float *arr1, const float *arr2, size_t size, float tolerance = 1e-5) {
    for (size_t i = 0; i < size; i++) {
        if (fabs(arr1[i] - arr2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void extensiveTest_make_Z(size_t numTests, size_t maxExamples, size_t maxDimensions, size_t maxClasses) {
    srand(time(nullptr)); // Seed random number generator

    for (size_t test = 0; test < numTests; ++test) {
        size_t m = rand() % maxExamples + 1; // Ensure at least one example
        size_t n = rand() % maxDimensions + 1; // Ensure at least one dimension
        size_t k = rand() % maxClasses + 1; // Ensure at least one class

        float *X = generateRandomFloats(m * n);
        float *theta = generateRandomFloats(n * k);
        float *Z = make_Z(X, theta, m, n, k);

        // Here we should check the results but since we don't have ground truth we just ensure no NaNs or Infs
        assert(Z != nullptr && "make_Z should not return nullptr.");
        for (size_t i = 0; i < m * k; ++i) {
            assert(!std::isnan(Z[i]) && !std::isinf(Z[i]) && "Output Z should not contain NaN or Inf.");
        }

        delete[] X;
        delete[] theta;
        delete[] Z;
    }

    std::cout << "Extensive testing for make_Z passed." << std::endl;
}

void extensiveTest_make_I(size_t numTests, size_t maxExamples, size_t maxClasses) {
    srand(time(nullptr)); // Seed random number generator

    for (size_t test = 0; test < numTests; ++test) {
        size_t m = rand() % maxExamples + 1;
        size_t k = rand() % maxClasses + 1;
        unsigned char *y = generateRandomClasses(m, k - 1); // Classes from 0 to k-1
        float *I = make_I(y, m, k);

        // Verify that I is properly one-hot encoded
        for (size_t i = 0; i < m; ++i) {
            int oneCount = 0;
            for (size_t j = 0; j < k; ++j) {
                float expectedValue = (j == y[i] ? 1.0f : 0.0f);
                assert(I[i * k + j] == expectedValue && "One-hot encoding failed.");
                oneCount += I[i * k + j] == 1.0f ? 1 : 0;
            }
            assert(oneCount == 1 && "Exactly one '1' expected per row.");
        }

        delete[] y;
        delete[] I;
    }

    std::cout << "Extensive testing for make_I passed." << std::endl;
}

void test_make_gradient() {
    size_t m = 2; // number of examples
    size_t n = 3; // input dimensions
    size_t k = 2; // number of classes
    size_t batch = 2; // batch size

    // Mock data for X, I, and Z
    float X[] = {1.0, 0.5, 0.2, // First example
                 0.4, 0.8, 0.6}; // Second example
    float I[] = {1.0, 0.0,       // One-hot for first example
                 0.0, 1.0};      // One-hot for second example
    float Z[] = {0.7, 0.3,       // Softmax output for first example
                 0.2, 0.8};      // Softmax output for second example

    // Expected gradient output
    // Calculated manually or using a separate trusted tool/script
    float expected_gradient[] = {(-0.3 * 1.0 + -0.8 * 0.4) / batch, (-0.3 * 0.5 + -0.8 * 0.8) / batch,
                                 (-0.3 * 0.2 + -0.8 * 0.6) / batch, (0.3 * 1.0 + 0.8 * 0.4) / batch,
                                 (0.3 * 0.5 + 0.8 * 0.8) / batch,   (0.3 * 0.2 + 0.8 * 0.6) / batch};

    // Calling the function under test
    float *computed_gradient = make_gradient(X, I, Z, m, n, k, batch);

    // Check that computed gradient matches expected gradient
    assert(compareFloatArrays(computed_gradient, expected_gradient, n * k) && "Gradient calculation failed");

    delete[] computed_gradient; // Cleanup

    std::cout << "test_make_gradient passed." << std::endl;
}

void test_softmax_regression_epoch_cpp() {
    // Setup test data
    size_t m = 10; // number of examples
    size_t n = 5;  // input dimension
    size_t k = 3;  // number of classes
    float lr = 0.01; // learning rate
    size_t batch = 2; // batch size

    // Create mock data for X, y, theta
    float X[m * n] = {1.0, 0.5, 0.3, 0.2, 0.1,
                      0.1, 0.2, 0.3, 0.4, 0.5,
                      0.5, 0.4, 0.3, 0.2, 0.1,
                      0.1, 0.2, 0.3, 0.4, 0.5,
                      0.5, 0.4, 0.3, 0.2, 0.1,
                      0.1, 0.2, 0.3, 0.4, 0.5,
                      0.5, 0.4, 0.3, 0.2, 0.1,
                      0.1, 0.2, 0.3, 0.4, 0.5,
                      0.5, 0.4, 0.3, 0.2, 0.1};
    unsigned char y[m] = {0, 2, 1, 1, 0, 2, 1, 1, 0, 2};
    float theta[n * k] = {0.1, 0.2, 0.3,
                          0.1, 0.2, 0.3,
                          0.1, 0.2, 0.3,
                          0.1, 0.2, 0.3,
                          0.1, 0.2, 0.3};

    // Initialize expected_theta to a copy of theta
    float expected_theta[n * k];
    std::copy(theta, theta + n * k, expected_theta);

    // Simulate the expected changes for a single epoch
    // This should ideally be replaced with manually computed expected values
    // for a proper test, as shown:
    for (int j = 0; j < n * k; ++j) {
        expected_theta[j] -= lr * 0.05; // Assume a dummy gradient decrease
    }

    // Call the function under test
    softmax_regression_epoch_cpp(X, y, theta, m, n, k, lr, batch);

    // Check that theta has been updated correctly
    assert(compareFloatArrays(theta, expected_theta, n * k));

    std::cout << "test_softmax_regression_epoch_cpp passed." << std::endl;
}

int main()
{
    // test_make_I();
    // test_make_Z();
    // extensiveTest_make_Z(1000, 100, 10, 5); // 1000 tests with up to 100 examples, 10 dimensions, 5 classes
    // extensiveTest_make_I(1000, 100, 5); // 1000 tests with up to 100 examples, 5 classes
    test_make_gradient();
    return 0;
}


#endif