#include <cstdint>
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


const size_t N = 2000000;

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
        k %= 1000'0000;

        v_set.insert(k);
    }
    auto tmp_v = vector<uint64_t> {};
    for (auto v: v_set) {
        tmp_v.push_back(v);
    }

    return tmp_v;
}

int test1() {
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    std::vector<std::pair<uint64_t, MarketData>> inputs;
    inputs.reserve(N);

    for (auto& v : get_random_int64_list(N)) {
        MarketData d;
        d.price = 1000.0 + v * 0.1;
        d.volume = v;
        std::strncpy(d.code, "BTC", 4);
        if (v == 0 || v == shm_pm::EMPTY_KEY) {
            v = 1;
        }
        inputs.push_back({v, d});
    }

    shm_pm::ShmMapStorage<MarketData, 4> storage;
    storage.build("test_pm1", inputs);
    auto& view = storage.get_view();

    auto lookups = get_random_int64_list(N);

    // 性能压测
    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0;
    for(auto i=0; i<10; i +=1) {
        for (auto k : lookups) {
            auto res = view.get(k);
            if (__builtin_expect(res->key == k, 1)) {
                found_cnt++;
                v += res->value.volume;
            }
        }
    }
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

    if (found_cnt != N) {
        std::cerr << "[Error] Data mismatch! Found " << found_cnt << ", expected " << N << std::endl;
        // 只有 Key 生成逻辑修正后，这里才不会报错
        return 1;
    }

    return 0;
}

int test3() {
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    auto u_map = ankerl::unordered_dense::map<uint64_t, MarketData>{};

    for (auto& v : get_random_int64_list(N)) {
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
    for(auto i=0; i<10; i +=1) {
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

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "v : " << std::to_string(v) << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (double) N / (elapsed_ns / 1e9) / 1e6 << " M ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

    if (found_cnt != N) {
        std::cerr << "[Error] Data mismatch! Found " << found_cnt << ", expected " << N << std::endl;
        // 只有 Key 生成逻辑修正后，这里才不会报错
        return 1;
    }

    return 0;
}

int main() {
    std::cout << "            test1             " << std::endl;
    test1();
    // std::cout << "            test2             "<< std::endl;
    // test2();
    std::cout << "            test3             " << std::endl;
    test3();
    return 0;
}