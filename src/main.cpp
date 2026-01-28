#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <array>
#include <random>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <cstdint>
#include <limits>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <set>
#include <atomic>
#include <thread>
#include <mutex>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "unordered_dense.h"

using namespace std;
using namespace ankerl;

template<uint64_t TableBits> class UltraCompactMapper {
    public:
        using KeyType = uint64_t;
        static constexpr size_t TableSize = 1ULL << TableBits;

        static constexpr uint64_t KEY_MASK_48 = 0x0000FFFFFFFFFFFFULL;
        static constexpr uint64_t ID_MASK_16 = 0xFFFF;

        alignas(64) KeyType table[TableSize];

        uint64_t magic_multiplier = 0;

        UltraCompactMapper() {
            memset(table, 0xFF, sizeof(KeyType) * TableSize);
        }

        __attribute__((always_inline)) inline KeyType load_key_asm(const uint8_t* src) const {
            uint64_t result;
            asm volatile("movq (%1), %0" : "=r"(result) : "r"(src) :);
            return result & KEY_MASK_48;
        }

        __attribute__((always_inline)) inline KeyType load_key_asm(const char* src) const {
            return this->load_key_asm(reinterpret_cast<const uint8_t*>(src));
        }

        __attribute__((always_inline)) int32_t get(const uint8_t* str) const {
            KeyType key = load_key_asm(str);
            auto idx = static_cast<uint32_t>((key * magic_multiplier) >> (64 - TableBits));

            KeyType entry = table[idx];
            auto stored_key = entry >> 16;

            return (key == stored_key) ? static_cast<int32_t>(static_cast<uint16_t>(entry & ID_MASK_16))
                                       : static_cast<int32_t>(-1);
        }

        __attribute__((always_inline)) int32_t get(const char* str) const {
            return this->get(reinterpret_cast<const uint8_t*>(str));
        }

        bool build(const vector<string>& known_keys, uint64_t max_attempts) {
            auto start_time = chrono::high_resolution_clock::now();

            vector<KeyType> nums;
            nums.reserve(known_keys.size());
            {
                for (const auto& k : known_keys) {
                    nums.push_back(load_key_asm(reinterpret_cast<const uint8_t*>(k.c_str())));
                }
                std::sort(nums.begin(), nums.end());
                nums.erase(std::unique(nums.begin(), nums.end()), nums.end());
            }

            if (nums.size() > TableSize) {
                cerr << "[Error] 数据量 (" << nums.size() << ") 超过表容量\n";
                return false;
            }

            uint64_t rng_state = 123456789ULL;
            auto fast_rand = [&rng_state]() -> uint64_t {
                uint64_t x = rng_state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                return rng_state = x;
            };

            vector<uint32_t> used(TableSize, 0);

            uint64_t attempts = 0;
            const uint32_t shift_amount = static_cast<uint32_t>(64 - TableBits);

            while (attempts < max_attempts) {
                attempts += 1;
                uint64_t candidate = fast_rand() | 1;
                bool collision = false;

                for (const auto k : nums) {
                    // index 计算
                    auto idx = static_cast<uint32_t>((k * candidate) >> shift_amount);

                    if (used[idx] == static_cast<uint32_t>(attempts)) {
                        collision = true;
                        break;
                    }
                    used[idx] = static_cast<uint32_t>(attempts);
                }

                if (!collision) {
                    magic_multiplier = candidate;

                    memset(table, 0xFF, sizeof(KeyType) * TableSize);
                    for (size_t i = 0; i < nums.size(); i += 1) {
                        KeyType k = nums[i];
                        auto idx = static_cast<uint32_t>((k * candidate) >> shift_amount);
                        table[idx] = (k << 16) | static_cast<KeyType>(i & ID_MASK_16);
                    }

                    auto end_time = chrono::high_resolution_clock::now();
                    auto elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();

                    double size_kb = (double) (sizeof(KeyType) * TableSize) / 1024.0;
                    double load_factor = 100.0 * (double) nums.size() / TableSize;

                    cout << "[Info] 完美哈希构建成功 (极速版):\n"
                         << "  - 表规格:   " << TableBits << " 位索引 (" << size_kb << " KB)\n"
                         << "  - 填充率:   " << nums.size() << "/" << TableSize << " (占比 " << fixed <<
                         setprecision(2)
                         << load_factor << "%)\n"
                         << "  - 魔数:     0x" << hex << uppercase << candidate << dec << "\n"
                         << "  - 尝试次数: " << attempts << "\n"
                         << "  - 总耗时:   " << elapsed_ms << " ms\n";
                    return true;
                }
            }

            auto end_time = chrono::high_resolution_clock::now();
            auto elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();
            auto size_kb = static_cast<double>(sizeof(KeyType) * TableSize) / 1024.0;
            auto load_factor = 100.0 * static_cast<double>(nums.size()) / static_cast<double>(TableSize);

            cerr << "[Error] 完美哈希构建失败:\n"
                 << "  - 表规格:   " << TableBits << " 位索引 (" << size_kb << " KB)\n"
                 << "  - 填充率:   " << nums.size() << "/" << TableSize << " (占比 " << fixed << setprecision(2)
                 << load_factor << "%)\n"
                 << "  - 已尝试:   " << attempts << " 次\n"
                 << "  - 总耗时:   " << elapsed_ms << " ms\n"
                 << "  - 建议:     请增加 TableBits 大小 (当前负载率过高)\n";

            return false;
        }


        // bool build(const vector<string>& known_keys, uint64_t max_attempts_total) {
        //     auto start_time = chrono::high_resolution_clock::now();

        //     vector<KeyType> nums;
        //     nums.reserve(known_keys.size());
        //     {
        //         for (const auto& k : known_keys) {
        //             nums.push_back(load_key_asm(reinterpret_cast<const uint8_t*>(k.c_str())));
        //         }
        //         std::sort(nums.begin(), nums.end());
        //         nums.erase(std::unique(nums.begin(), nums.end()), nums.end());
        //     }

        //     if (nums.size() > TableSize) {
        //         cerr << "[Error] 数据量 (" << nums.size() << ") 超过表容量\n";
        //         return false;
        //     }

        //     unsigned int thread_count = std::thread::hardware_concurrency();
        //     if (thread_count == 0) {
        //         thread_count = 1;
        //     }

        //     std::atomic<bool> found(false);       // 全局标志位：是否有人找到了
        //     std::atomic<uint64_t> total_tries(0); // 统计总尝试次数
        //     uint64_t winning_candidate = 0;       // 存储结果
        //     std::mutex result_mutex;              // 保护写入

        //     uint64_t attempts_per_thread = max_attempts_total / thread_count;

        //     auto worker = [&](int thread_id) {
        //         uint64_t rng_state = 123456789ULL + thread_id * 9999999;
        //         auto fast_rand = [&rng_state]() -> uint64_t {
        //             uint64_t x = rng_state;
        //             x ^= x << 13;
        //             x ^= x >> 7;
        //             x ^= x << 17;
        //             return rng_state = x;
        //         };

        //         vector<uint32_t> used(TableSize, 0);
        //         uint32_t cookie = 0; // 对应 used 中的值

        //         const uint32_t shift_amount = static_cast<uint32_t>(64 - TableBits);
        //         uint64_t local_attempts = 0;

        //         while (local_attempts < attempts_per_thread && !found.load(std::memory_order_relaxed)) {
        //             local_attempts++;
        //             cookie++;

        //             if (cookie == 0) {
        //                 cookie = 1;
        //                 std::memset(used.data(), 0, sizeof(uint16_t) * TableSize);
        //             }

        //             uint64_t candidate = fast_rand() | 1;
        //             bool collision = false;

        //             for (const auto k : nums) {
        //                 auto idx = static_cast<uint32_t>((k * candidate) >> shift_amount);

        //                 if (used[idx] == cookie) {
        //                     collision = true;
        //                     break;
        //                 }
        //                 used[idx] = static_cast<uint32_t>(cookie);
        //             }

        //             if (!collision) {
        //                 if (!found.exchange(true)) {
        //                     std::lock_guard<std::mutex> lock(result_mutex);
        //                     winning_candidate = candidate;
        //                 }
        //                 break;
        //             }

        //             if ((local_attempts & 0xFFF) == 0) {
        //                 total_tries.fetch_add(0x1000, std::memory_order_relaxed);
        //             }
        //         }
        //         total_tries.fetch_add(local_attempts % 0x1000, std::memory_order_relaxed);
        //     };

        //     vector<std::thread> threads;
        //     cout << "[Info] 启动 " << thread_count << " 个线程并行搜索..." << endl;
        //     for (unsigned int i = 0; i < thread_count; ++i) {
        //         threads.emplace_back(worker, i);
        //     }

        //     for (auto& t : threads)
        //         t.join();

        //     if (found) {
        //         magic_multiplier = winning_candidate;

        //         memset(table, 0xFF, sizeof(KeyType) * TableSize);
        //         const uint32_t shift_amount = static_cast<uint32_t>(64 - TableBits);
        //         for (size_t i = 0; i < nums.size(); ++i) {
        //             KeyType k = nums[i];
        //             auto idx = static_cast<uint32_t>((k * winning_candidate) >> shift_amount);
        //             table[idx] = (k << 16) | static_cast<KeyType>(i & ID_MASK_16);
        //         }

        //         auto end_time = chrono::high_resolution_clock::now();
        //         auto elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();
        //         double size_kb = (double) (sizeof(KeyType) * TableSize) / 1024.0;
        //         double load_factor = 100.0 * (double) nums.size() / TableSize;

        //         cout << "[Info] 完美哈希构建成功:\n"
        //              << "  - 表规格:   " << TableBits << " 位 (" << size_kb << " KB)\n"
        //              << "  - 填充率:   " << nums.size() << "/" << TableSize << " (" << fixed << setprecision(2)
        //              << load_factor << "%)\n"
        //              << "  - 总尝试:   " << total_tries << " 次 (并发)\n"
        //              << "  - 总耗时:   " << elapsed_ms << " ms\n";
        //         return true;
        //     } else {
        //         cerr << "[Error] 构建失败，尝试次数耗尽。建议增加 TableBits。\n";
        //         return false;
        //     }
        // }
};

vector<string> generate_random_keys(size_t count, size_t length = 6) {
    static constexpr char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    constexpr size_t charset_len = sizeof(charset) - 1;
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<size_t> dist(0, charset_len - 1);

    unordered_set<string> unique_set;
    vector<string> keys;

    keys.reserve(count);
    unique_set.reserve(count);

    while (keys.size() < count) {
        string s;
        s.reserve(length);

        for (size_t i = 0; i < length; i += 1) {
            s += charset[dist(rng)];
        }

        if (unique_set.insert(s).second) {
            keys.push_back(move(s));
        }
    }

    return keys;
}

vector<string> generate_simple_keys(size_t count) {
    vector<string> keys;
    for (int i = 0; i < 5000; ++i) {
        char b[8];
        sprintf(b, "%06d", i);
        keys.push_back(b);
    }
    return keys;
}

template <typename T>
inline void do_not_optimize(T const& val) {
    asm volatile("" : : "g"(val) : "memory");
}

int main() {
    // auto keys = generate_simple_keys(2000);
    auto keys = generate_random_keys(2000, 6);

    constexpr auto table_bits = 19; // 超过20 会有段错误
    auto mapper = UltraCompactMapper<table_bits>{};
    // auto mapper = std::make_unique<UltraCompactMapper<20>>();
    mapper.build(keys, 2'0000'0000);


    uint32_t aux;
    uint64_t start = __rdtscp(&aux);
    for(auto& k2:keys) {
        do_not_optimize(mapper.get(k2.c_str()));
    }
    uint64_t end = __rdtscp(&aux);
    cout << end - start << endl;



    auto u_map = unordered_dense::set<string>{};
    for(auto& k2:keys) {
        u_map.emplace(move(k2));
    }

    start = __rdtscp(&aux);
    for(auto& k2:keys) {
        do_not_optimize(u_map.find(k2));
    }
    end = __rdtscp(&aux);
    cout << end - start << endl;

    return 0;
}