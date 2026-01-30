// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cstring>
// #include <cstdint>
// #include <random>
// #include <chrono>
// #include <type_traits>
// #include <stdexcept>
// #include <iomanip>

// // -----------------------------------------------------------------------------
// // 高性能哈希混合器 (基于 WyHash / Murmur 变体)
// // -----------------------------------------------------------------------------
// namespace HashCore {
//     // 快速混合函数，用于将 Key 和 Seed 混合生成索引
//     // 使用乘法和异或位移，延迟低且雪崩效应好
//     __attribute__((always_inline)) inline uint64_t mix(uint64_t key, uint64_t seed) {
//         key ^= seed;
//         key *= 0xa0761d6478bd642fULL;
//         key ^= key >> 47;
//         key *= 0xe7037ed1a0b428dbULL;
//         key ^= key >> 47;
//         return key;
//     }

//     // Fast Range: 将 64位哈希值映射到 [0, range) 区间
//     // 比取模 (%) 快得多，使用定点乘法原理
//     __attribute__((always_inline)) inline uint32_t fast_map(uint64_t hash, uint32_t range) {
//         return static_cast<uint32_t>((static_cast<unsigned __int128>(hash) * range) >> 64);
//     }
// }

// // -----------------------------------------------------------------------------
// // 扁平化完美哈希表 (Flat Perfect Hash Map)
// // -----------------------------------------------------------------------------
// // T: 存储的值类型，必须是 POD (Plain Old Data)，以便存入共享内存
// template<typename T>
// class FlatPerfectMap {
// public:
//     static_assert(std::is_trivially_copyable_v<T>, "为了实现共享内存兼容性, T必须是POD");

//     struct Entry {
//         uint64_t key;
//         T value;
//     };

//     // 内存头信息 (64字节对齐)
//     struct Header {
//         uint32_t bucket_count;     // 第一级桶的数量
//         uint32_t data_capacity;    // 数据总量
//         uint64_t total_size_bytes; // 整个内存块的大小
//         uint64_t magic;            // 校验魔数
//         uint8_t  padding[40];      // 补齐 Cache Line
//     };

// private:
//     const uint8_t* base_ptr = nullptr;      // 内存基地址
//     const Header* header = nullptr;         // 头信息
//     const int32_t* control_table = nullptr; // 控制表 (存储 seed)
//     const Entry* data_table = nullptr;      // 数据表 (存储实际数据)

// public:
//     FlatPerfectMap() = default;

//     // -------------------------------------------------------------------------
//     // 加载接口: 传入一块内存 (来自 mmap, vector 或共享内存)
//     // -------------------------------------------------------------------------
//     void load_from_memory(const void* ptr, size_t size) {
//         base_ptr = reinterpret_cast<const uint8_t*>(ptr);
//         header = reinterpret_cast<const Header*>(base_ptr);

//         if (header->magic != 0xFA57F1A7) {
//             throw std::runtime_error("Invalid FlatPerfectMap magic signature");
//         }
//         if (header->total_size_bytes > size) {
//             throw std::runtime_error("Buffer too small");
//         }

//         // 布局计算: [Header] [Control Table] [Padding] [Data Table]
//         size_t control_offset = sizeof(Header);
//         control_table = reinterpret_cast<const int32_t*>(base_ptr + control_offset);

//         // 确保 Data Table 起始地址 64 字节对齐，利用 SIMD 加载优势
//         size_t control_size = header->bucket_count * sizeof(int32_t);
//         size_t data_offset = control_offset + control_size;
//         size_t remainder = data_offset % 64;
//         if (remainder != 0) data_offset += (64 - remainder);

//         data_table = reinterpret_cast<const Entry*>(base_ptr + data_offset);
//     }

//     // -------------------------------------------------------------------------
//     // 核心查找 (Hot Path) - 极致性能
//     // -------------------------------------------------------------------------
//     // 返回值指针，未找到返回 nullptr
//     __attribute__((always_inline)) const T* get(uint64_t key) const {
//         // 1. 第一级哈希: 定位到 Bucket (使用常量 seed 预混淆)
//         uint64_t h1 = HashCore::mix(key, 0x1234567890ABCDEFULL);
//         uint32_t bucket_idx = HashCore::fast_map(h1, header->bucket_count);

//         // 2. 查控制表: 获取该 Bucket 的 Displacement Seed
//         // 这个访问极快，因为 Control Table 很小，通常在 L1/L2 Cache 中
//         int32_t seed = control_table[bucket_idx];

//         // 3. 快速路径: Seed < 0 表示该桶为空
//         if (seed < 0) return nullptr;

//         // 4. 第二级哈希: 利用 Seed 定位到唯一 Slot
//         // 这里没有循环，没有探测，直接计算出物理偏移
//         uint64_t h2 = HashCore::mix(key, static_cast<uint64_t>(seed));
//         uint32_t slot_idx = HashCore::fast_map(h2, header->data_capacity);

//         // 5. 访问数据 (可能发生 L3 Cache Miss，但只发生一次)
//         const Entry& entry = data_table[slot_idx];

//         // 6. 最终校验: 完美哈希只能保证已知 Key 无冲突，
//         // 对于未知 Key 可能会映射到同一位置，必须校验 Key 是否相等
//         if (entry.key == key) {
//             return &entry.value;
//         }

//         return nullptr;
//     }

//     // -------------------------------------------------------------------------
//     // 构建器 (Builder) - 离线运行
//     // -------------------------------------------------------------------------
//     static std::vector<uint8_t> build(const std::vector<std::pair<uint64_t, T>>& input_data) {
//         size_t n = input_data.size();
//         if (n == 0) return {};

//         // 1. 初始化桶
//         // 桶的数量通常设为 N 的一部分，甚至可以 > N 以降低构建难度
//         // 这里设为 N * 1.25，兼顾空间和构建速度
//         uint32_t num_buckets = static_cast<uint32_t>(n * 1.25);
//         if (num_buckets < 4) num_buckets = 4;

//         std::vector<std::vector<size_t>> buckets(num_buckets);

//         // 2. 第一步: 将所有 Key 分配到桶中
//         for (size_t i = 0; i < n; ++i) {
//             uint64_t key = input_data[i].first;
//             uint64_t h1 = HashCore::mix(key, 0x1234567890ABCDEFULL);
//             uint32_t b_idx = HashCore::fast_map(h1, num_buckets);
//             buckets[b_idx].push_back(i); // 记录原始数据的索引
//         }

//         // 3. 桶排序: 必须先处理元素最多的桶 (关键启发式策略)
//         // 这样难处理的桶先占据 Slot，小桶填空缝
//         struct BucketRef { uint32_t id; size_t size; };
//         std::vector<BucketRef> refs(num_buckets);
//         for(uint32_t i=0; i<num_buckets; ++i) refs[i] = {i, buckets[i].size()};

//         std::sort(refs.begin(), refs.end(), [](const auto& a, const auto& b){
//             return a.size > b.size; // 降序
//         });

//         // 4. 分配 Slot 和寻找 Seed
//         std::vector<int32_t> control(num_buckets, -1);
//         std::vector<bool> slot_occupied(n, false);
//         std::vector<Entry> final_data(n);

//         // 使用固定的随机数序列，保证构建过程确定性
//         std::mt19937 rng(999);

//         for (const auto& ref : refs) {
//             if (ref.size == 0) continue;

//             uint32_t b_idx = ref.id;
//             const auto& item_indices = buckets[b_idx];

//             // 暴力搜索该桶的 Seed
//             bool seed_found = false;

//             // 尝试 1000 万次，通常只要几百次就能找到
//             // 这里的 seed 只需要是 int32
//             for (int32_t seed = 0; seed < 10000000; ++seed) {
//                 // 如果觉得顺序搜索慢，可以用 rng() 生成随机 seed，但顺序搜索对 Cache 更友好

//                 bool collision = false;
//                 std::vector<uint32_t> pending_slots;
//                 pending_slots.reserve(ref.size);

//                 for (size_t original_idx : item_indices) {
//                     uint64_t k = input_data[original_idx].first;
//                     // 使用当前 seed 试探位置
//                     uint64_t h2 = HashCore::mix(k, static_cast<uint64_t>(seed));
//                     uint32_t slot = HashCore::fast_map(h2, static_cast<uint32_t>(n));

//                     // 检查全局冲突 (是否已被其他桶占用)
//                     if (slot_occupied[slot]) { collision = true; break; }

//                     // 检查桶内冲突 (本桶内两个key映射到了同一个slot)
//                     for(uint32_t s : pending_slots) if(s == slot) { collision = true; break; }
//                     if(collision) break;

//                     pending_slots.push_back(slot);
//                 }

//                 if (!collision) {
//                     // 找到了完美的 Seed!
//                     control[b_idx] = seed;
//                     for (size_t i = 0; i < ref.size; ++i) {
//                         uint32_t slot = pending_slots[i];
//                         size_t original_idx = item_indices[i];

//                         slot_occupied[slot] = true;
//                         final_data[slot] = {input_data[original_idx].first, input_data[original_idx].second};
//                     }
//                     seed_found = true;
//                     break;
//                 }
//             }

//             if (!seed_found) {
//                 std::cerr << "[Error] Build failed for bucket " << b_idx << " size=" << ref.size << "\n";
//                 return {};
//             }
//         }

//         // 5. 序列化 (Flattening)
//         size_t header_sz = sizeof(Header);
//         size_t ctrl_sz = num_buckets * sizeof(int32_t);
//         size_t data_sz = n * sizeof(Entry);

//         // 计算对齐填充
//         size_t offset_before_data = header_sz + ctrl_sz;
//         size_t padding = 0;
//         if (offset_before_data % 64 != 0) padding = 64 - (offset_before_data % 64);

//         size_t total_size = header_sz + ctrl_sz + padding + data_sz;
//         std::vector<uint8_t> buffer(total_size);
//         uint8_t* ptr = buffer.data();

//         // 写入 Header
//         Header h;
//         h.bucket_count = num_buckets;
//         h.data_capacity = static_cast<uint32_t>(n);
//         h.total_size_bytes = total_size;
//         h.magic = 0xFA57F1A7;
//         memset(h.padding, 0, sizeof(h.padding));
//         memcpy(ptr, &h, sizeof(Header));

//         // 写入 Control Table
//         memcpy(ptr + header_sz, control.data(), ctrl_sz);

//         // 写入 Data Table
//         memcpy(ptr + header_sz + ctrl_sz + padding, final_data.data(), data_sz);

//         std::cout << "[Info] Perfect Hash Built: " << n << " keys. Size: "
//                   << total_size / 1024.0 << " KB. Efficiency: "
//                   << (double)(n * sizeof(Entry)) / total_size * 100.0 << "%\n";

//         return buffer;
//     }
// };

// // =============================================================================
// // 测试代码 & Benchmark
// // =============================================================================

// // 为了禁用优化，精准测量
// template <typename T>
// inline void do_not_optimize(T const& val) {
//     asm volatile("" : : "g"(val) : "memory");
// }

// // 示例 Value 类型 (POD)
// struct MarketData {
//     double price;
//     uint32_t volume;
//     char code[4];
// };

// int main() {
//     // 1. 生成测试数据
//     const size_t N = 50'0000; // 100万个 Key
//     std::vector<std::pair<uint64_t, MarketData>> inputs;
//     inputs.reserve(N);

//     std::mt19937_64 rng(123);
//     for(size_t i=0; i<N; ++i) {
//         MarketData d;
//         d.price = i * 0.1;
//         d.volume = i;
//         memcpy(d.code, "ETH", 4);
//         inputs.push_back({rng(), d}); // 随机 Key
//     }

//     // 2. 构建 (模拟离线过程)
//     auto start_build = std::chrono::high_resolution_clock::now();
//     auto buffer = FlatPerfectMap<MarketData>::build(inputs);
//     auto end_build = std::chrono::high_resolution_clock::now();
//     std::cout << "Build time: " << std::chrono::duration<double, std::milli>(end_build - start_build).count() << " ms\n";

//     // 3. 加载 (模拟在线过程)
//     FlatPerfectMap<MarketData> map;
//     map.load_from_memory(buffer.data(), buffer.size());

//     // 4. 性能压测 (Lookup Benchmark)
//     // 混洗查询顺序，模拟真实随机访问，制造 Cache Miss
//     std::vector<uint64_t> lookups;
//     lookups.reserve(N);
//     for(auto& p : inputs) lookups.push_back(p.first);
//     std::shuffle(lookups.begin(), lookups.end(), rng);

//     auto start_lookup = std::chrono::high_resolution_clock::now();

//     uint64_t found_cnt = 0;
//     for(uint64_t k : lookups) {
//         const MarketData* res = map.get(k);
//         if(res) {
//             found_cnt++;
//             do_not_optimize(res->price);
//         }
//     }

//     auto end_lookup = std::chrono::high_resolution_clock::now();
//     auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

//     std::cout << "Lookup Count: " << N << "\n";
//     std::cout << "Found: " << found_cnt << "\n";
//     std::cout << "Total Time: " << elapsed_ns / 1000000.0 << " ms\n";
//     std::cout << "Latency per op: " << (double)elapsed_ns / N << " ns (Amazing!)\n";

//     return 0;
// }


#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <random>
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <immintrin.h> // SIMD
#include <sys/mman.h>  // 为了使用 madvise/hugepages


namespace HashCore {
    __attribute__((always_inline)) inline uint64_t hash_one_pass(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }
}

template<typename T>
class ExtremePerfectMap {
public:
    static_assert(std::is_trivially_copyable_v<T>, "T must be POD");

    struct Header {
        uint64_t bucket_mask;      // Control Table 的掩码 (size - 1)
        uint64_t slot_mask;        // Data Table 的掩码 (size - 1)
        uint32_t bucket_shift;     // 用于提取高位: hash >> bucket_shift
        uint32_t padding;
        uint64_t total_size_bytes;
        uint64_t magic;
    };

private:
    const uint8_t* base_ptr = nullptr;
    const Header* header = nullptr;
    const uint64_t* control_table = nullptr; // 存储 64位 Seed
    const uint64_t* key_table = nullptr;     // SoA: 纯 Key 数组
    const T* value_table = nullptr;          // SoA: 纯 Value 数组

    static uint64_t next_pow2(uint64_t x) {
        if (x == 0) return 1;
        x--;
        x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
        return x + 1;
    }

public:
    ExtremePerfectMap() = default;

    void load_from_memory(const void* ptr, size_t size) {
        base_ptr = reinterpret_cast<const uint8_t*>(ptr);
        header = reinterpret_cast<const Header*>(base_ptr);

        madvise(const_cast<void*>(ptr), size, MADV_HUGEPAGE | MADV_WILLNEED);

        if (header->magic != MAGIC_CODE) {
            throw std::runtime_error("Invalid magic");
        }

        if (header->total_size_bytes > size) {
            throw std::runtime_error("Buffer too small");
        }

        auto align64 = [](size_t s) { return (s + 63) & ~63; };

        size_t offset_ctrl = sizeof(Header);
        size_t ctrl_sz     = (header->bucket_mask + 1) * sizeof(uint64_t);

        size_t offset_keys = align64(offset_ctrl + ctrl_sz);
        size_t key_sz      = (header->slot_mask + 1) * sizeof(uint64_t);

        size_t offset_vals = align64(offset_keys + key_sz);

        control_table = reinterpret_cast<const uint64_t*>(base_ptr + offset_ctrl);
        key_table     = reinterpret_cast<const uint64_t*>(base_ptr + offset_keys);
        value_table   = reinterpret_cast<const T*>(base_ptr + offset_vals);
    }


    __attribute__((always_inline)) const T* get(uint64_t key) const {
        uint64_t h = HashCore::hash_one_pass(key);
        uint64_t bucket_idx = h >> header->bucket_shift;
        uint64_t seed = control_table[bucket_idx];
        uint64_t slot_idx = (h ^ seed) & header->slot_mask;

        uint64_t stored_key = key_table[slot_idx];
        if (__builtin_expect(stored_key == key, 1)) {
            return &value_table[slot_idx];
        }
        return nullptr;
    }

    static std::vector<uint8_t> build(const std::vector<std::pair<uint64_t, T>>& data) {
        size_t n = data.size();
        if (n == 0) return {};

        double slot_factor = 2.0;  // 初始两倍空间
        double bucket_factor = 0.8; // 桶的密度因子

        while (true) {
            uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
            if (slot_cnt < 1024) slot_cnt = 1024;

            uint64_t bucket_cnt = next_pow2(static_cast<uint64_t>(n * bucket_factor));
            if (bucket_cnt < 4) bucket_cnt = 4;

            std::cout << "[Build] Trying... Keys:" << n
                      << " Buckets:" << bucket_cnt
                      << " Slots:" << slot_cnt
                      << " (Load Factor: " << std::fixed << std::setprecision(1)
                      << (double)n/slot_cnt*100 << "%)" << std::endl;

            // 1. 准备掩码
            uint64_t bucket_mask = bucket_cnt - 1;
            uint64_t slot_mask = slot_cnt - 1;
            int bucket_bits = 0;
            while ((1ULL << bucket_bits) < bucket_cnt) {
                bucket_bits +=1;
            };
            uint32_t bucket_shift = 64 - bucket_bits;

            // 2. 分桶
            struct BucketInfo {
                uint64_t id;
                std::vector<uint64_t> keys;
            };
            std::vector<BucketInfo> buckets(bucket_cnt);
            for(uint64_t i=0; i<bucket_cnt; ++i) {
                buckets[i].id = i;
            };

            for (const auto& kv : data) {
                uint64_t h = HashCore::hash_one_pass(kv.first);
                uint64_t b_idx = h >> bucket_shift;
                buckets[b_idx].keys.push_back(kv.first);
            }

            // 3. 排序 (大桶优先)
            std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b){
                return a.keys.size() > b.keys.size();
            });

            // 4. 尝试寻找 Seed
            std::vector<uint64_t> control(bucket_cnt, 0);
            std::vector<bool> slot_used(slot_cnt, false);
            std::vector<uint64_t> final_keys(slot_cnt, 0);
            std::vector<T> final_values(slot_cnt);

            std::mt19937_64 rng(123456); // 固定种子方便调试，或者用 random_device
            bool build_success = true;

            for (const auto& bucket : buckets) {
                if (bucket.keys.empty()) continue;

                bool found = false;
                // 尝试次数
                int max_attempts = 2000000;

                for (int attempt = 0; attempt < max_attempts; ++attempt) {
                    uint64_t seed = rng();
                    bool collision = false;
                    std::vector<uint64_t> proposed_slots;
                    proposed_slots.reserve(bucket.keys.size());

                    for (uint64_t k : bucket.keys) {
                        uint64_t h = HashCore::hash_one_pass(k);
                        uint64_t s_idx = (h ^ seed) & slot_mask;

                        if (slot_used[s_idx]) { collision = true; break; }
                        for(auto ps : proposed_slots) if(ps == s_idx) { collision = true; break; }
                        if(collision) break;
                        proposed_slots.push_back(s_idx);
                    }

                    if (!collision) {
                        control[bucket.id] = seed;
                        for (size_t i = 0; i < bucket.keys.size(); ++i) {
                            uint64_t s_idx = proposed_slots[i];
                            slot_used[s_idx] = true;
                            final_keys[s_idx] = bucket.keys[i];

                            // 线性查找 value (为了简化代码)
                            for(const auto& input_kv : data) {
                                if(input_kv.first == bucket.keys[i]) {
                                    final_values[s_idx] = input_kv.second;
                                    break;
                                }
                            }
                        }
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    build_success = false;
                    break; // 当前 bucket 失败，直接放弃这一轮
                }
            }

            if (build_success) {
                // --- 构建成功，开始序列化 ---
                size_t header_sz = sizeof(Header);
                size_t ctrl_sz = bucket_cnt * sizeof(uint64_t);
                size_t key_sz = slot_cnt * sizeof(uint64_t);
                size_t val_sz = slot_cnt * sizeof(T);
                auto align64 = [](size_t s) { return (s + 63) & ~63; };

                size_t offset_ctrl = header_sz;
                size_t offset_keys = align64(offset_ctrl + ctrl_sz);
                size_t offset_vals = align64(offset_keys + key_sz);
                size_t total_sz = offset_vals + val_sz;

                std::vector<uint8_t> buffer(total_sz);
                uint8_t* base = buffer.data();
                Header* h = reinterpret_cast<Header*>(base);
                h->bucket_mask = bucket_mask;
                h->slot_mask = slot_mask;
                h->bucket_shift = bucket_shift;
                h->total_size_bytes = total_sz;
                h->magic = MAGIC_CODE;

                memcpy(base + offset_ctrl, control.data(), ctrl_sz);
                memcpy(base + offset_keys, final_keys.data(), key_sz);
                memcpy(base + offset_vals, final_values.data(), val_sz);

                std::cout << "[Build] Success! Final Load Factor: "
                          << (double)n/slot_cnt*100 << "%" << std::endl;
                return buffer;

            } else {
                std::cout << "[Build] Retry: Increasing capacity..." << std::endl;
                slot_factor *= 1.5;
                bucket_factor *= 1.2;

                if (slot_factor > 20.0) {
                    std::cerr << "[Fatal] Cannot build perfect hash even with huge space." << std::endl;
                    return {};
                }
            }
        }
    }

    static const uint64_t MAGIC_CODE = 0x8899AABBCCDDEEFF;
};





// #include "unordered_dense.h"

// int main() {
//     auto u_map = ankerl::unordered_dense::map<uint64_t, MarketData>{};

//     const size_t N = 50'0000; // 测试规模: 200万个 Key
//     std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

//     std::vector<std::pair<uint64_t, MarketData>> inputs;
//     inputs.reserve(N);

//     // 使用固定种子的随机数生成器，保证每次 Benchmark 数据一致
//     std::mt19937_64 rng(12345);

//     for(size_t i = 0; i < N; ++i) {
//         MarketData d;
//         d.price = 1000.0 + i * 0.1;
//         d.volume = i;
//         std::strncpy(d.code, "BTC", 4);

//         // 生成随机 Key (模拟 64位 OrderID 或 UserID)
//         uint64_t key = rng();
//         // 保证 Key 不为 0 (视具体实现而定，通常 0 留作空位标记)
//         if (key == 0) key = 1;

//         u_map[key] = d;
//         inputs.push_back({key, d});
//     }

//     // ---------------------------------------------------------
//     // 步骤 2: 构建阶段 (模拟“离线”生成数据文件)
//     // ---------------------------------------------------------
//     std::cout << "[Build] Building ExtremePerfectMap..." << std::endl;
//     auto start_build = std::chrono::high_resolution_clock::now();

//     // 调用构建函数 (注意: 之前我们修改了内部参数以空间换时间)
//     auto buffer = ExtremePerfectMap<MarketData>::build(inputs);

//     auto end_build = std::chrono::high_resolution_clock::now();
//     double build_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();

//     if (buffer.empty()) {
//         std::cerr << "[Fatal] Build failed! Please increase slot_cnt multiplier." << std::endl;
//         return 1;
//     }

//     std::cout << "  - Build Time: " << build_ms << " ms" << std::endl;
//     std::cout << "  - Buffer Size: " << buffer.size() / 1024.0 / 1024.0 << " MB" << std::endl;

//     // ---------------------------------------------------------
//     // 步骤 3: 加载阶段 (模拟“在线”服务启动)
//     // ---------------------------------------------------------
//     ExtremePerfectMap<MarketData> map;
//     try {
//         // 这里模拟从 mmap 或者共享内存读取
//         map.load_from_memory(buffer.data(), buffer.size());
//     } catch (const std::exception& e) {
//         std::cerr << "[Fatal] Load failed: " << e.what() << std::endl;
//         return 1;
//     }

//     // ---------------------------------------------------------
//     // 步骤 4: 性能压测 (Lookup Benchmark)
//     // ---------------------------------------------------------
//     std::cout << "[Bench] Starting Lookup Benchmark (Randomized/Cache-Miss)..." << std::endl;

//     // 准备查询 Key 列表
//     std::vector<uint64_t> lookups;
//     lookups.reserve(N);
//     for(const auto& p : inputs) {
//         lookups.push_back(p.first);
//     }

//     std::shuffle(lookups.begin(), lookups.end(), rng);

//     // 预热 (可选，让 OS 分配物理页)
//     map.get(lookups[0]);

//     auto start_lookup = std::chrono::high_resolution_clock::now();

//     uint64_t found_cnt = 0;

//     // 核心测试循环
//     for(uint64_t k : lookups) {
//         const MarketData* res = map.get(k);

//         // 使用 unlikely 提示编译器，虽然我们知道一定会命中，
//         // 但我们要测试的是分支预测逻辑是否影响流水线
//         if (__builtin_expect(res != nullptr, 1)) {
//             found_cnt++;
//             // 访问内存，确保 Cache Line 被加载
//             do_not_optimize(res->price);
//         }

//         // auto res = u_map.find(k);
//         // if (__builtin_expect(res != u_map.end(), 1)) {
//         //     found_cnt++;
//         //     do_not_optimize(res->second.price);
//         // }
//     }

//     auto end_lookup = std::chrono::high_resolution_clock::now();
//     auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

//     // ---------------------------------------------------------
//     // 步骤 5: 结果报告
//     // ---------------------------------------------------------
//     double latency = (double)elapsed_ns / N;

//     std::cout << "========================================" << std::endl;
//     std::cout << "Lookup Count : " << N << std::endl;
//     std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
//     std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
//     std::cout << "Latency/Op   : " << latency << " ns " << (latency < 10.0 ? "(Extremely Fast!)" : "") << std::endl;
//     std::cout << "========================================" << std::endl;

//     // 校验完整性
//     if (found_cnt != N) {
//         std::cerr << "[Error] Data mismatch! Lost keys." << std::endl;
//         return 1;
//     }

//     return 0;
// }


#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "lib.hpp" // 假设之前的头文件保存为 hash_storage.hpp
#include "unordered_dense.h"

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

int test1() {

    // 0. 清理旧环境 (确保测试每次都是全新构建)
    shm_unlink("test_pm1");

    const size_t N = 50'0000; // 规模: 100万，更能体现 Cache Miss
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    std::vector<std::pair<uint64_t, MarketData>> inputs;
    inputs.reserve(N);

    std::mt19937_64 rng(12345);

    for (size_t i = 0; i < N; ++i) {
        MarketData d;
        d.price = 1000.0 + i * 0.1;
        d.volume = i;
        std::strncpy(d.code, "BTC", 4);

        uint64_t key = rng();
        if (key == 0 || key == ~0ULL)
            key = 1; // 避开 0 和 EMPTY_KEY
        inputs.push_back({key, d});
    }

    // ---------------------------------------------------------
    // 步骤 2: 构建阶段
    // ---------------------------------------------------------
    std::cout << "[Build] Building Perfect Hash Map..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();

    // [修正 1] ALIGN 参数
    // MarketData 包含 double，alignof 为 8。你之前传入 4 是非法操作(Undefined Behavior)。
    // 建议使用 64 (Cache Line) 以避免多线程 False Sharing，或者 16 (紧凑布局)。
    // 这里为了 Benchmark 极致读取性能，我们使用 16 (sizeof MarketData)，
    // 注意：如果有多线程写，必须用 64。
    shm_pm::ShmMapStorage<MarketData, 4> storage;

    try {
        storage.build("test_pm1", inputs);
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] Build failed: " << e.what() << std::endl;
        return 1;
    }

    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "[Build] Done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() << " ms."
              << std::endl;

    // ---------------------------------------------------------
    // 步骤 3: 准备查询数据
    // ---------------------------------------------------------
    std::cout << "[Bench] Preparing Lookup keys (Randomized)..." << std::endl;
    std::vector<uint64_t> lookups;
    lookups.reserve(N);
    for (const auto& p : inputs) {
        lookups.push_back(p.first);
    }
    // 打乱顺序，模拟真实的随机访问 (强制 Cache Miss)
    std::shuffle(lookups.begin(), lookups.end(), rng);

    auto& view = storage.get_view();

    // ---------------------------------------------------------
    // [修正 2] Warm-up (预热) - 可选
    // ---------------------------------------------------------
    // Linux mmap 是惰性加载的 (Page Fault)。
    // 如果你想测“纯粹的内存查询算法耗时”，需要预热。
    // 如果你想测“系统冷启动后第一次访问耗时”，则不要预热。
    // 这里我们做一次预热，排除 OS Page Fault 的干扰，测纯算法性能。
    {
        volatile double sum = 0;
        for (uint64_t i = 0; i < N; i += 1000) { // 简单跳跃访问，触发 TLB/Page载入
            auto* ptr = view.get(lookups[i]);
            sum += ptr->price;
        }
    }

    // ---------------------------------------------------------
    // 步骤 4: 性能压测
    // ---------------------------------------------------------
    std::cout << "[Bench] Starting Measurement..." << std::endl;

    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0; // Accumulator

    for (uint64_t k : lookups) {
        // [核心路径]
        const MarketData* res = view.get(k);

        found_cnt++;
        // 累加操作比 do_not_optimize 更接近真实业务逻辑(读取字段)
        v += res->price;
    }
    // 防止循环被优化
    do_not_optimize(v);

    auto end_lookup = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

    // ---------------------------------------------------------
    // 步骤 5: 报告
    // ---------------------------------------------------------
    double latency = (double) elapsed_ns / N;

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "Check Sum    : " << std::fixed << std::setprecision(2) << v << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (long) ((double) N / (elapsed_ns / 1e9)) / 1000000.0 << " M ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

    if (found_cnt != N) {
        std::cerr << "[Error] Data mismatch! Found " << found_cnt << " expected " << N << std::endl;
        return 1;
    }

    return 0;
}


int test2() {
    const size_t N = 50'0000; // 规模: 100万，更能体现 Cache Miss
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    std::vector<std::pair<uint64_t, MarketData>> inputs;
    inputs.reserve(N);

    std::mt19937_64 rng(12345);

    for (size_t i = 0; i < N; ++i) {
        MarketData d;
        d.price = 1000.0 + i * 0.1;
        d.volume = i;
        std::strncpy(d.code, "BTC", 4);

        uint64_t key = rng();
        if (key == 0 || key == ~0ULL)
            key = 1; // 避开 0 和 EMPTY_KEY
        inputs.push_back({key, d});
    }

    // ---------------------------------------------------------
    // 步骤 2: 构建阶段
    // ---------------------------------------------------------
    std::cout << "[Build] Building Perfect Hash Map..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();

    ExtremePerfectMap<MarketData> map;
    auto buffer = ExtremePerfectMap<MarketData>::build(inputs);
    map.load_from_memory(buffer.data(), buffer.size());

    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "[Build] Done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() << " ms."
              << std::endl;

    // ---------------------------------------------------------
    // 步骤 3: 准备查询数据
    // ---------------------------------------------------------
    std::cout << "[Bench] Preparing Lookup keys (Randomized)..." << std::endl;
    std::vector<uint64_t> lookups;
    lookups.reserve(N);
    for (const auto& p : inputs) {
        lookups.push_back(p.first);
    }
    // 打乱顺序，模拟真实的随机访问 (强制 Cache Miss)
    std::shuffle(lookups.begin(), lookups.end(), rng);


    // ---------------------------------------------------------
    // [修正 2] Warm-up (预热) - 可选
    // ---------------------------------------------------------
    // Linux mmap 是惰性加载的 (Page Fault)。
    // 如果你想测“纯粹的内存查询算法耗时”，需要预热。
    // 如果你想测“系统冷启动后第一次访问耗时”，则不要预热。
    // 这里我们做一次预热，排除 OS Page Fault 的干扰，测纯算法性能。
    {
        volatile double sum = 0;
        for (uint64_t i = 0; i < N; i += 1000) { // 简单跳跃访问，触发 TLB/Page载入
            auto* ptr = map.get(lookups[i]);
            sum += ptr->price;
        }
    }

    // ---------------------------------------------------------
    // 步骤 4: 性能压测
    // ---------------------------------------------------------
    std::cout << "[Bench] Starting Measurement..." << std::endl;

    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0; // Accumulator

    for (uint64_t k : lookups) {
        // [核心路径]
        const MarketData* res = map.get(k);

        found_cnt++;
        // 累加操作比 do_not_optimize 更接近真实业务逻辑(读取字段)
        v += res->price;
    }
    // 防止循环被优化
    do_not_optimize(v);

    auto end_lookup = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

    // ---------------------------------------------------------
    // 步骤 5: 报告
    // ---------------------------------------------------------
    double latency = (double) elapsed_ns / N;

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "Check Sum    : " << std::fixed << std::setprecision(2) << v << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (long) ((double) N / (elapsed_ns / 1e9)) / 1000000.0 << " M ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

    if (found_cnt != N) {
        std::cerr << "[Error] Data mismatch! Found " << found_cnt << " expected " << N << std::endl;
        return 1;
    }

    return 0;
}


int test3() {
    const size_t N = 50'0000; // 规模: 100万，更能体现 Cache Miss
    std::cout << "[Init] Generating " << N << " random keys..." << std::endl;

    std::vector<std::pair<uint64_t, MarketData>> inputs;
    inputs.reserve(N);

    std::mt19937_64 rng(12345);

    for (size_t i = 0; i < N; ++i) {
        MarketData d;
        d.price = 1000.0 + i * 0.1;
        d.volume = i;
        std::strncpy(d.code, "BTC", 4);

        uint64_t key = rng();
        if (key == 0 || key == ~0ULL)
            key = 1; // 避开 0 和 EMPTY_KEY
        inputs.push_back({key, d});
    }

    // ---------------------------------------------------------
    // 步骤 2: 构建阶段
    // ---------------------------------------------------------
    std::cout << "[Build] Building Perfect Hash Map..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();

    auto map = ankerl::unordered_dense::map<uint64_t, MarketData>{};

    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "[Build] Done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() << " ms."
              << std::endl;

    // ---------------------------------------------------------
    // 步骤 3: 准备查询数据
    // ---------------------------------------------------------
    std::cout << "[Bench] Preparing Lookup keys (Randomized)..." << std::endl;
    std::vector<uint64_t> lookups;
    lookups.reserve(N);
    for (const auto& p : inputs) {
        map[p.first] = p.second;
        lookups.push_back(p.first);
    }
    // 打乱顺序，模拟真实的随机访问 (强制 Cache Miss)
    std::shuffle(lookups.begin(), lookups.end(), rng);


    // ---------------------------------------------------------
    // [修正 2] Warm-up (预热) - 可选
    // ---------------------------------------------------------
    // Linux mmap 是惰性加载的 (Page Fault)。
    // 如果你想测“纯粹的内存查询算法耗时”，需要预热。
    // 如果你想测“系统冷启动后第一次访问耗时”，则不要预热。
    // 这里我们做一次预热，排除 OS Page Fault 的干扰，测纯算法性能。
    {
        volatile double sum = 0;
        for (uint64_t i = 0; i < N; i += 1000) { // 简单跳跃访问，触发 TLB/Page载入
            auto ptr = map.find(lookups[i]);
            if(ptr != map.end()) {
                sum += ptr->second.price;
            }
        }
    }

    // ---------------------------------------------------------
    // 步骤 4: 性能压测
    // ---------------------------------------------------------
    std::cout << "[Bench] Starting Measurement..." << std::endl;

    auto start_lookup = std::chrono::high_resolution_clock::now();

    uint64_t found_cnt = 0;
    double v = 0.0; // Accumulator

    for (uint64_t k : lookups) {
        // [核心路径]
        auto res = &map[k];

        found_cnt++;
        v += res->price;
    }
    // 防止循环被优化
    do_not_optimize(v);

    auto end_lookup = std::chrono::high_resolution_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_lookup - start_lookup).count();

    // ---------------------------------------------------------
    // 步骤 5: 报告
    // ---------------------------------------------------------
    double latency = (double) elapsed_ns / N;

    std::cout << "========================================" << std::endl;
    std::cout << "Lookup Count : " << N << std::endl;
    std::cout << "Check Sum    : " << std::fixed << std::setprecision(2) << v << std::endl;
    std::cout << "Hit Count    : " << found_cnt << " / " << N << std::endl;
    std::cout << "Total Time   : " << elapsed_ns / 1000000.0 << " ms" << std::endl;
    std::cout << "Latency/Op   : " << latency << " ns " << std::endl;
    std::cout << "Throughput   : " << (long) ((double) N / (elapsed_ns / 1e9)) / 1000000.0 << " M ops/sec" << std::endl;
    std::cout << "========================================" << std::endl;

    if (found_cnt != N) {
        std::cerr << "[Error] Data mismatch! Found " << found_cnt << " expected " << N << std::endl;
        return 1;
    }

    return 0;
}

int main() {
    std::cout << "            test1             "<< std::endl;
    test1();
    std::cout << "            test2             "<< std::endl;
    test2();
    std::cout << "            test3             "<< std::endl;
    test3();
    return 0;
}