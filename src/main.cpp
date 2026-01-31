#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "lib.hpp" // 假设之前的头文件保存为 hash_storage.hpp
#include "unordered_dense.h"

using namespace std;


static auto N = 200000;

// ---------------------------------------------------------
// 辅助函数：防止编译器优化掉读操作
// ---------------------------------------------------------
template<typename T>
inline void do_not_optimize(T const& val) {
    asm volatile("" : : "g"(val) : "memory");
}

struct MarketData {
        double price;    // 8 bytes
        uint32_t volume; // 4 bytes
        char code[4];    // 4 bytes
}; // Total: 16 bytes, alignof: 8

std::vector<uint64_t> get_random_int64_list(size_t n) {
    auto v_set = ankerl::unordered_dense::set<uint64_t>{};
    std::mt19937_64 rng(std::random_device{}());

    while (v_set.size() < n) {
        uint64_t k = rng();
        if (k == 0 || k == shm_pm::EMPTY_KEY)
            continue;
        k %= 10'0000'0000;

        v_set.insert(k);
    }
    auto tmp_v = vector<uint64_t>{};
    for (auto v : v_set) {
        tmp_v.push_back(v);
    }

    return tmp_v;
}

int test1() {
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    std::vector<std::pair<uint64_t, MarketData>> inputs;
    inputs.reserve(N);

    auto random_lis = get_random_int64_list(N);
    for (auto& v : random_lis) {
        MarketData d;
        d.price = 1000.0 + v * 0.1;
        d.volume = v;
        std::strncpy(d.code, "BTC", 4);
        if (v == 0 || v == shm_pm::EMPTY_KEY) {
            v = 1;
        }
        inputs.push_back({v, d});
    }

    shm_pm::ShmMapStorage<MarketData, uint8_t, 8> storage;
    storage.build("test_pm1", inputs);
    auto& view = storage.get_view();

    auto lookups = get_random_int64_list(N);

    // 性能压测
    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0;
    for (auto i = 0; i < 10; i += 1) {
        for (auto k : lookups) {
            auto res = view.get(k);
            if (__builtin_expect(res->key == k, 1)) {
                found_cnt++;
                v += res->value.volume;
            }
        }
    }
    cout << view.capacity() << "   " << view.size() << endl;

    // for (auto i = 0; i < 10; i += 1) {
    //     for (auto& res: view) {
    //         found_cnt++;
    //         v += res.value.volume;
    //     }
    //     // for (size_t k = 0; k < N; k += 1) {

    //     //     auto res = view.at(k);

    //     // }
    // }
    do_not_optimize(v);

    auto end_lookup = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

    // 报告
    double latency = (double) elapsed_ns / N / 10;

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "v : " << std::to_string(v) << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (double) N / (elapsed_ns / 1e9) / 1e6 << " M ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}

int test3() {
    struct HighPerfCustomHash {
            using is_avalanching = void;

            auto operator()(std::string_view const& s) const noexcept -> uint64_t {
                uint64_t hash = 0xcbf29ce484222325;            // FNV_offset_basis
                constexpr uint64_t prime = 0x1099511628211900; // FNV_prime

                const char* ptr = s.data();
                size_t len = s.size();

                while (len > 0) {
                    uint8_t byte = static_cast<uint8_t>(*ptr);
                    hash = hash ^ byte;
                    hash = hash * prime;

                    ptr = ptr + 1;
                    len = len - 1;
                }
                return hash;
            }

            auto operator()(uint64_t x) const noexcept -> uint64_t {
                return x * 11400714819323198485ULL;
            }
    };

    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    auto u_map = ankerl::unordered_dense::map<uint64_t, MarketData, HighPerfCustomHash>{};

    auto random_lis = get_random_int64_list(N);
    for (auto& v : random_lis) {
        MarketData d;
        d.price = 1000.0 + v * 0.1;
        d.volume = v;
        std::strncpy(d.code, "BTC", 4);
        if (v == 0 || v == shm_pm::EMPTY_KEY) {
            v = 1;
        }
        u_map[v] = d;
    }

    auto lookups = get_random_int64_list(N);

    // 性能压测
    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0;
    for (auto i = 0; i < 10; i += 1) {
        for (uint64_t k : lookups) {
            auto p = u_map.find(k);
            if (p != u_map.end()) {
                found_cnt++;
                v += p->second.volume;
            }
        }
    }
    do_not_optimize(v);

    auto end_lookup = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

    // 报告
    double latency = (double) elapsed_ns / N / 10;

    using MapType = decltype(u_map);
    size_t bucket_bytes = u_map.bucket_count() * sizeof(typename MapType::bucket_type);
    size_t value_bytes = u_map.values().capacity() * sizeof(typename MapType::value_type);

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "v : " << std::to_string(v) << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (double) N / (elapsed_ns / 1e9) / 1e6 << " M ops/sec" << std::endl;


    std::cout << "Memory Report:\n";
    std::cout << "  Buckets (Index Overhead): " << (bucket_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Values  (Actual Data):    " << (value_bytes / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Total Est. Memory:        " << ((bucket_bytes + value_bytes) / 1024.0 / 1024.0) << " MB\n";
    std::cout << "========================================" << std::endl;

    return 0;
}

int main(const int argc, char* argv[]) {
    if (argc < 2) {
        throw runtime_error("请传入: 元素数量");
    }

    N = atoi(argv[1]);

    std::cout << "            test1             " << std::endl;
    test1();
    // std::cout << "            test2             "<< std::endl;
    // test2();
    std::cout << "            test3             " << std::endl;
    test3();
    return 0;
}