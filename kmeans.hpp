#pragma once

#include "emmintrin.h"
#include "immintrin.h"

#include <array>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>

#ifdef _WIN32
#pragma comment ( lib, "Shlwapi.lib" )
#ifdef _DEBUG
#pragma comment ( lib, "benchmark.lib" )
#pragma comment ( lib, "benchmark_main.lib" )
#endif
#endif 
/* If you are using MinGW on Windows, it is possible that your compiler does not support <thread>.
 * We have posted solution to such case on Piazza. Please read the post @124.
 */

using index_t = int;

struct Point
{
    double x, y;

    Point() : x(0), y(0) {};
    Point(double x_, double y_) : x(x_), y(y_) {}
    Point(const Point& other) = default;

    inline double Distance(const Point& other) const;
};

constexpr int thread_count = 16;
constexpr double inf = 1e100;

class Kmeans
{
public:
    Kmeans(const std::vector<Point>& points, const std::vector<Point>& initialCenters);
    ~Kmeans();
    std::vector<index_t> Run(int maxIterations=1000);

private:
    __m256d* m_points_x;
    __m256d* m_points_y;
    __m256d* m_centers_x;
    __m256d* m_centers_y;
    double* m_copied_points_x;
    double* m_copied_points_y;
    double* m_copied_centers_x;
    double* m_copied_centers_y;
    int m_numPoints;
    int m_numPointsWithPadding;
    int m_numCenters;
    int m_numCentersWithPadding;

    void RefreshAverage(const std::array<std::vector<std::pair<Point,int>>,thread_count>& temp);

    __inline static __m256d  FastDistance4(__m256d x1_4,__m256d x2_4, __m256d y1_4, __m256d y2_4){
        auto dx = _mm256_sub_pd(x1_4,x2_4);
        auto dy = _mm256_sub_pd(y1_4,y2_4);
        return _mm256_add_pd(_mm256_mul_pd(dx,dx),_mm256_mul_pd(dy,dy));
    }

    void PointClassify(std::vector<std::pair<Point, int>> &point, index_t *assignment, int begin, int end);

    __inline static void GetMinDistance4Of4(__m256d px, __m256d py, __m256d cx, __m256d cy, __m256i indices, __m256i &min_indices, __m256d &min_values);
};

std::istream& operator>>(std::istream& is, Point& pt);
std::ostream& operator<<(std::ostream& os, Point& pt);