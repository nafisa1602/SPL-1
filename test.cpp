#include <iostream>
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
}
