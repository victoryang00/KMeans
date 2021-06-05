#include "kmeans.hpp"

inline double Point::Distance(const Point &other) const {
    return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
}

std::istream &operator>>(std::istream &is, Point &pt) {
    return is >> pt.x >> pt.y;
}

std::ostream &operator<<(std::ostream &os, Point &pt) {
    return os << pt.x << ' ' << pt.y;
}


Kmeans::Kmeans(const std::vector<Point> &points, const std::vector<Point> &initialCenters) {
    m_numPoints = int(points.size());
    if (m_numPoints % 4 == 0) {
        m_numPointsWithPadding = m_numPoints;
    } else {
        m_numPointsWithPadding = (m_numPoints / 4 + 1) * 4;
    }
    m_numCenters = int(initialCenters.size());
    if (m_numCenters % 4 == 0) {
        m_numCentersWithPadding = m_numCenters;
    } else {
        m_numCentersWithPadding = (m_numCenters / 4 + 1) * 4;
    }
    m_points_x = (__m256d *) _mm_malloc(m_numPointsWithPadding * sizeof(double), 32);
    m_points_y = (__m256d *) _mm_malloc(m_numPointsWithPadding * sizeof(double), 32);
    m_centers_x = (__m256d *) _mm_malloc(m_numCentersWithPadding * sizeof(double), 32);
    m_centers_y = (__m256d *) _mm_malloc(m_numCentersWithPadding * sizeof(double), 32);
    m_copied_points_x = (double *) _mm_malloc(m_numPointsWithPadding * sizeof(double), 32);
    m_copied_points_y = (double *) _mm_malloc(m_numPointsWithPadding * sizeof(double), 32);
    m_copied_centers_x = (double *) _mm_malloc(m_numCentersWithPadding * sizeof(double), 32);
    m_copied_centers_y = (double *) _mm_malloc(m_numCentersWithPadding * sizeof(double), 32);
    {
        const Point *point_ptr = points.data();
        int i;
        for (i = 0; i < m_numPoints; ++i) {
            m_copied_points_x[i] = (point_ptr + i)->x;
            m_copied_points_y[i] = (point_ptr + i)->y;
        }
        for (i = m_numPoints; i < m_numPointsWithPadding; ++i) {
            m_copied_points_x[i] = inf;
            m_copied_points_y[i] = inf;
        }
    }
    {
        const Point *point_ptr = initialCenters.data();
        int i;
        for (i = 0; i < m_numCenters; ++i) {
            m_copied_centers_x[i] = (point_ptr + i)->x;
            m_copied_centers_y[i] = (point_ptr + i)->y;
        }
        for (i = m_numCenters; i < m_numCentersWithPadding; ++i) {
            m_copied_centers_x[i] = inf;
            m_copied_centers_y[i] = inf;
        }
    }
    {
        int i;
        int size = m_numPointsWithPadding / 4;
        for (i = 0; i < size; ++i) {
            m_points_x[i] = _mm256_load_pd(m_copied_points_x + 4 * i);
            m_points_y[i] = _mm256_load_pd(m_copied_points_y + 4 * i);
        }
    }
    {
        int i;
        int size = m_numCentersWithPadding / 4;
        for (i = 0; i < size; ++i) {
            m_centers_x[i] = _mm256_load_pd(m_copied_centers_x + i * 4);
            m_centers_y[i] = _mm256_load_pd(m_copied_centers_y + i * 4);
        }
    }
}

std::vector<index_t> Kmeans::Run(int maxIterations) {
    // DO NOT MODIFY THESE CODE
    std::vector<index_t> assignment(m_numPoints, 0); // the return vector
    int currIteration = 0;
    std::cout << "Running kmeans with num points = " << m_numPoints
              << ", num centers = " << m_numCenters
              << ", max iterations = " << maxIterations << "...\n";

    // YOUR CODE HERE
    for (; currIteration < maxIterations; ++currIteration) {
        std::array<std::vector<std::pair<Point, int>>, thread_count> temp; //sum of points
        for (auto &vec:temp) {
            vec = std::vector<std::pair<Point, int>>(m_numCenters, std::make_pair(Point(), 0));
        }
        auto assignment_old = assignment;

        std::vector<std::thread> threads;
        const int slice_size = m_numPointsWithPadding / 4 / thread_count;
        for (int i = 0; i < thread_count; ++i) {
            if (i != thread_count - 1) {
                threads.emplace_back(&Kmeans::PointClassify, this, std::ref(temp[i]), assignment.data(),
                                     slice_size * 4 * i, slice_size * 4 * (i + 1));
            } else {
                threads.emplace_back(&Kmeans::PointClassify, this, std::ref(temp[i]), assignment.data(),
                                     slice_size * 4 * i, m_numPoints);
            }
        }
        for (int i = 0; i < thread_count; ++i) {
            threads[i].join();
        }
        if (assignment_old == assignment) {
            break;
        }
        RefreshAverage(temp);
    }
    // YOUR CODE ENDS HERE
    std::cout << "Finished in " << currIteration << " iterations." << std::endl;
    return assignment;
}

__inline void Kmeans::GetMinDistance4Of4(__m256d px, __m256d py, __m256d cx, __m256d cy, __m256i indices, __m256i& min_indices, __m256d& min_values) {
    constexpr int compare_option = _CMP_LT_OS;

    min_indices = indices;
    min_values = FastDistance4(px, cx, py, cy);

    indices = _mm256_permute4x64_epi64(indices, 0b00111001);
    cx = _mm256_permute4x64_pd(cx, 0b00111001);
    cy = _mm256_permute4x64_pd(cy, 0b00111001);

    __m256d values = FastDistance4(px, cx, py, cy);
    __m256d lt = _mm256_cmp_pd(values, min_values, compare_option);
    min_indices = _mm256_blendv_epi8(min_indices, indices, _mm256_castpd_si256(lt));
    min_values = _mm256_blendv_pd(min_values, values, lt);

    indices = _mm256_permute4x64_epi64(indices, 0b00111001);
    cx = _mm256_permute4x64_pd(cx, 0b00111001);
    cy = _mm256_permute4x64_pd(cy, 0b00111001);

    values = FastDistance4(px, cx, py, cy);
    lt = _mm256_cmp_pd(values, min_values, compare_option);
    min_indices = _mm256_blendv_epi8(min_indices, indices, _mm256_castpd_si256(lt));
    min_values = _mm256_blendv_pd(min_values, values, lt);

    indices = _mm256_permute4x64_epi64(indices, 0b00111001);
    cx = _mm256_permute4x64_pd(cx, 0b00111001);
    cy = _mm256_permute4x64_pd(cy, 0b00111001);

    values = FastDistance4(px, cx, py, cy);
    lt = _mm256_cmp_pd(values, min_values, compare_option);
    min_indices = _mm256_blendv_epi8(min_indices, indices, _mm256_castpd_si256(lt));
    min_values = _mm256_blendv_pd(min_values, values, lt);
}

void
Kmeans::PointClassify(std::vector<std::pair<Point, int>> &point, index_t *assignment, const int begin, const int end) {
    constexpr int compare_option = _CMP_LT_OS;
    const auto center_count = m_numCentersWithPadding / 4;
    auto *point_x_agg = (double *) _mm_malloc(m_numCenters * sizeof(double), 32);
    auto *point_y_agg = (double *) _mm_malloc(m_numCenters * sizeof(double), 32);
    int *point_hit_count = (int *) _mm_malloc(m_numCenters * sizeof(int), 32);
    memset(point_x_agg, 0, m_numCenters * sizeof(double));
    memset(point_y_agg, 0, m_numCenters * sizeof(double));
    memset(point_hit_count, 0, m_numCenters * sizeof(int));
    const __m256i increment = _mm256_set1_epi64x(4);
    auto it = begin/4;
    for (; it < end / 4; it++) {
        __m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i min_indices;
        __m256d min_values;
        GetMinDistance4Of4(m_points_x[it],m_points_y[it],m_centers_x[0],m_centers_y[0],indices,min_indices,min_values);
        for (auto center_idx = 1; center_idx < center_count; ++center_idx) {
            indices = _mm256_add_epi64(indices, increment);
            __m256i new_min_indices;
            __m256d new_min_values;
            GetMinDistance4Of4(m_points_x[it],m_points_y[it],m_centers_x[center_idx],m_centers_y[center_idx],indices,new_min_indices,new_min_values);
            __m256d lt = _mm256_cmp_pd(new_min_values, min_values, compare_option);
            min_indices = _mm256_blendv_epi8(min_indices, new_min_indices, _mm256_castpd_si256(lt));
            min_values = _mm256_blendv_pd(min_values, new_min_values,lt);
        }

        auto ptr = (long long*)(&min_indices);
        assignment[it*4] = int(ptr[0]);
        point_x_agg[ptr[0]] += ((double *) (&(m_points_x[it])))[0];
        point_y_agg[ptr[0]] += ((double *) (&(m_points_y[it])))[0];
        ++point_hit_count[ptr[0]];

        assignment[it*4+1] = int(ptr[1]);
        point_x_agg[ptr[1]] += ((double *) (&(m_points_x[it])))[1];
        point_y_agg[ptr[1]] += ((double *) (&(m_points_y[it])))[1];
        ++point_hit_count[ptr[1]];

        assignment[it*4+2] = int(ptr[2]);
        point_x_agg[ptr[2]] += ((double *) (&(m_points_x[it])))[2];
        point_y_agg[ptr[2]] += ((double *) (&(m_points_y[it])))[2];
        ++point_hit_count[ptr[2]];

        assignment[it*4+3] = int(ptr[3]);
        point_x_agg[ptr[3]] += ((double *) (&(m_points_x[it])))[3];
        point_y_agg[ptr[3]] += ((double *) (&(m_points_y[it])))[3];
        ++point_hit_count[ptr[3]];
    }

    if(it*4 < end){
        __m256i indices = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i min_indices;
        __m256d min_values;
        GetMinDistance4Of4(m_points_x[it],m_points_y[it],m_centers_x[0],m_centers_y[0],indices,min_indices,min_values);
        for (auto center_idx = 1; center_idx < center_count; ++center_idx) {
            indices = _mm256_add_epi64(indices, increment);
            __m256i new_min_indices;
            __m256d new_min_values;
            GetMinDistance4Of4(m_points_x[it],m_points_y[it],m_centers_x[center_idx],m_centers_y[center_idx],indices,new_min_indices,new_min_values);
            __m256d lt = _mm256_cmp_pd(new_min_values, min_values, compare_option);
            min_indices = _mm256_blendv_epi8(min_indices, new_min_indices, _mm256_castpd_si256(lt));
            min_values = _mm256_blendv_pd(min_values, new_min_values,lt);
        }

        auto ptr = (long long*)(&min_indices);
        for(auto i=0;i<end-it*4;++i){
            assignment[it*4+i] = int(ptr[i]);
            point_x_agg[ptr[i]] += ((double *) (&(m_points_x[it])))[i];
            point_y_agg[ptr[i]] += ((double *) (&(m_points_y[it])))[i];
            ++point_hit_count[ptr[i]];
        }
    }

    for (int i = 0; i < m_numCenters; ++i) {
        point[i].first.x = point_x_agg[i];
        point[i].first.y = point_y_agg[i];
        point[i].second = point_hit_count[i];
    }
    _mm_free(point_x_agg);
    _mm_free(point_y_agg);
    _mm_free(point_hit_count);
}

inline void Kmeans::RefreshAverage(const std::array<std::vector<std::pair<Point, int>>, thread_count> &temp) {
    for (int i = 0; i < m_numCenters; ++i) {
        m_copied_centers_x[i] = 0;
        m_copied_centers_y[i] = 0;
    }
    std::vector<int> total_count(m_numCenters, 0);
    for (const auto &vec:temp) {
        for (int i = 0; i < m_numCenters; ++i) {
            m_copied_centers_x[i] += vec[i].first.x;
            m_copied_centers_y[i] += vec[i].first.y;
            total_count[i] += vec[i].second;
        }
    }
    for (int i = 0; i < m_numCenters; ++i) {
        m_copied_centers_x[i] /= double(total_count[i]);
        m_copied_centers_y[i] /= double(total_count[i]);
    }
    {
        int i;
        int size = m_numCentersWithPadding / 4;
        for (i = 0; i < size; ++i) {
            m_centers_x[i] = _mm256_load_pd(m_copied_centers_x + i * 4);
            m_centers_y[i] = _mm256_load_pd(m_copied_centers_y + i * 4);
        }
    }
}

Kmeans::~Kmeans() {
    _mm_free(m_centers_x);
    _mm_free(m_centers_y);
    _mm_free(m_points_x);
    _mm_free(m_points_y);
    _mm_free(m_copied_centers_x);
    _mm_free(m_copied_centers_y);
    _mm_free(m_copied_points_x);
    _mm_free(m_copied_points_y);
}


int main(int argc, char** argv) {
    // Seed with a real random value, if available
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(1, 6);
    int mean = uniform_dist(e1);
    std::cout << "Randomly-chosen mean: " << mean << '\n';

    // Generate a normal distribution around that mean
    std::seed_seq seed2{ r(), r(), r(), r(), r(), r(), r(), r() };
    std::mt19937 e2(seed2);
    std::normal_distribution<> normal_dist(mean, 2);

    int T = 1000000;
    std::vector<Point> all_points,init_mid;
    // Running K-Means Clustering
    int iters = 100;

    for (int iter = 0; iter < 100000; ++iter) {
        double random1 = std::round(normal_dist(e2));
        double random2 = std::round(normal_dist(e2));
        Point point(random1,random2);
        all_points.push_back(point);
        if (iter%10 == 0) {
            init_mid.push_back(point);
        }
    }

    Kmeans kmeans(all_points,init_mid);

    auto start = std::chrono::high_resolution_clock::now();
    kmeans.Run(iters);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "elapsed nanoseconds per elements: " << elapsed / T << std::endl;
    std::cerr << elapsed / T << std::endl;
    return 0;
}   