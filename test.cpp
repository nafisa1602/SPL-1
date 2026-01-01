/*#include <iostream>
#include "basic_math.h"
#include "advanced_math.h"
using namespace std;
using namespace basic_math;
using namespace advanced_math;

int main()
{
    cout << "=== Testing abs() ===\n";
    cout << "abs(-5.5) = " << abs(-5.5) << "\n";
    cout << "abs(3.2)  = " << abs(3.2)  << "\n\n";

    cout << "=== Testing min() and max() ===\n";
    cout << "min(2.5, 7.1) = " << min(2.5, 7.1) << "\n";
    cout << "max(2.5, 7.1) = " << max(2.5, 7.1) << "\n\n";

    cout << "=== Testing factorial() ===\n";
    cout << "factorial(5) = " << factorial(5) << "\n\n";

    cout << "=== Testing power() ===\n";
    cout << "power(2, 5)  = " << power(2.0, 5) << "\n";
    cout << "power(2,-3)  = " << power(2.0, -3) << "\n";
    cout << clamp(10, 20, 50) << endl;
    return 0;
}*/
#include <iostream>
#include "basic_math.h"
#include "advanced_math.h"
#include "vector_math.h"
#include "matrix_math.h"

int main()
{
    using namespace basic_math;
    using namespace advanced_math;
    using namespace vector_math;
    using namespace matrix_math;

    std::cout << "Power(2,10): " << power(2,10) << std::endl;
    std::cout << "Factorial(5): " << factorial(5) << std::endl;

    double v[3] = {1,2,3};
    double w[3] = {4,5,6};
    double out[3];

    vectorAddition(v, w, out, 3);
    std::cout << "Vector add: ";
    for(double x : out) std::cout << x << " ";
    std::cout << std::endl;

    double A[6] = {1,2,3,4,5,6};
    double B[6] = {7,8,9,10,11,12};
    double C[4];

    if(matrixMultiply(A, B, 2, 3, 3, 2, C))
    {
        std::cout << "Matrix multiply:\n";
        matrixPrint(C, 2, 2);
    }

    double soft[3];
    softMax(v, soft, 3);
    std::cout << "Softmax: ";
    for(double x : soft) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}

