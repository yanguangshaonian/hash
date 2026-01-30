#ifndef HASH_STORAGE_HPP
#define HASH_STORAGE_HPP
#pragma pack(push)
#pragma pack()

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <thread>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cerrno>

namespace shm_pm {
    using namespace std;
    using namespace std::chrono;

    // ----------------------------------------------------------------
    // 基础常量与工具
    // ----------------------------------------------------------------
    constexpr size_t CACHE_LINE_SIZE = 64;
    constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024; // 2MB
    constexpr size_t MIN_MAP_SIZE = 4096;              // 用于读取Header的最小页
    constexpr uint64_t MAGIC_CODE = 0xDEADBEEF20260130;
    constexpr uint64_t EMPTY_KEY = ~0ULL;

    inline uint64_t align_to_huge_page(uint64_t size) {
        return (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    }

    inline uint64_t align_to_cache_line(uint64_t size) {
        return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    }

    namespace HashCore {
        // 高性能混合哈希 (Murmur/WyHash 变体)
        __attribute__((always_inline)) inline uint64_t hash_one_pass(uint64_t k) {
            k ^= k >> 33;
            k *= 0xff51afd7ed558ccdULL;
            k ^= k >> 33;
            k *= 0xc4ceb9fe1a85ec53ULL;
            k ^= k >> 33;
            return k;
        }
    } // namespace HashCore

    // ----------------------------------------------------------------
    // 并发安全的数据单元 (自旋锁保护)
    // ----------------------------------------------------------------
    template<class T, size_t ALIGN = CACHE_LINE_SIZE>
    class alignas(ALIGN) PaddedValue {
        public:
            T value;
            std::atomic<bool> busy_flag = false;

            // 指数退避自旋锁
            inline __attribute__((always_inline)) bool lock(bool allow_fail = true) noexcept {
                while (true) {
                    uint8_t delay = 0b0000'0001;
                    while (busy_flag.load(std::memory_order_relaxed)) {
                        for (auto i = 0; i < delay; i += 1) {
                            asm volatile("pause" ::: "memory");
                        }

                        if (delay < 0b1000'0000) {
                            delay <<= 1;
                        } else {
                            if (allow_fail) {
                                return false;
                            }
                        }
                    }

                    if (!busy_flag.exchange(true, std::memory_order_acquire)) {
                        return true;
                    }
                }
            }

            inline __attribute__((always_inline)) void unlock() noexcept {
                busy_flag.store(false, std::memory_order_release);
            }
    };

    // ----------------------------------------------------------------
    // 共享内存头部 (Header)
    // ----------------------------------------------------------------
    class alignas(CACHE_LINE_SIZE) ShmMapHeader {
        public:
            volatile uint64_t magic; // 初始化完成标志

            // 完美哈希参数
            uint64_t bucket_mask;
            uint64_t slot_mask;
            uint32_t bucket_shift;

            // 统计信息
            uint64_t item_count;
            uint64_t bucket_count;
            uint64_t slot_count;

            // 内存偏移量 (相对于基地址)
            uint64_t offset_control;
            uint64_t offset_keys;
            uint64_t offset_values;

            // 文件元数据
            uint64_t total_file_size;
            uint64_t align_param;       // 记录 ALIGN
            uint64_t value_size;        // 记录 sizeof(T)
            uint64_t padded_value_size; // 记录 sizeof(PaddedValue)
    };

    // ----------------------------------------------------------------
    // 映射视图 (View) - 负责数据访问逻辑
    // ----------------------------------------------------------------
    template<class T, size_t ALIGN = CACHE_LINE_SIZE>
    class SharedMapView {
        private:
            const uint8_t* base_ptr = nullptr;
            const ShmMapHeader* header = nullptr;

            const uint64_t* control_table = nullptr;
            const uint64_t* key_table = nullptr;
            PaddedValue<T, ALIGN>* value_table = nullptr;

        public:
            void init(uint8_t* mapped_address) {
                base_ptr = mapped_address;
                header = reinterpret_cast<const ShmMapHeader*>(base_ptr);

                // 此时假设 Header 已经校验过 Magic 和 版本
                control_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_control);
                key_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_keys);
                value_table = reinterpret_cast<PaddedValue<T, ALIGN>*>(mapped_address + header->offset_values);
            }

            __attribute__((always_inline)) const T* get(uint64_t key) const {
                uint64_t h = HashCore::hash_one_pass(key);
                uint64_t bucket_idx = h >> header->bucket_shift;
                uint64_t seed = control_table[bucket_idx];
                uint64_t slot_idx = (h ^ seed) & header->slot_mask;

                if (__builtin_expect(key_table[slot_idx] == key, 1)) {
                    return &value_table[slot_idx].value;
                }
                return nullptr;
            }

            size_t capacity() const {
                return header ? header->slot_count : 0;
            }

            size_t size() const {
                return header ? header->item_count : 0;
            }
    };

    // ----------------------------------------------------------------
    // 存储管理器
    // ----------------------------------------------------------------
    template<class T, size_t ALIGN = CACHE_LINE_SIZE>
    class ShmMapStorage {
        private:
            enum class JoinResult {
                SUCCESS,
                FILE_NOT_FOUND, // 需要去创建
                DATA_CORRUPT,   // 文件存在但无效 (Magic 错误或正在初始化)
                TYPE_MISMATCH,  // 数据结构版本不一致
                SYSTEM_ERROR    // mmap 失败等系统级错误
            };

            int shm_fd = -1;
            uint8_t* mapped_ptr = nullptr;
            uint64_t mapped_size = 0;
            std::string storage_name;
            SharedMapView<T, ALIGN> view;

            void log_msg(const std::string& level, const std::string& msg) {
                auto now = system_clock::now();
                auto in_time_t = system_clock::to_time_t(now);
                std::tm bt{};
                localtime_r(&in_time_t, &bt);
                std::cout << "[" << std::put_time(&bt, "%T") << "] "
                          << "[ShmMap][" << level << "] "
                          << "[" << storage_name << "] " << msg << std::endl;
            }

            uint8_t* map_memory_segment(size_t size, bool use_hugepage) {
                int flags = MAP_SHARED;
                if (use_hugepage) {
                    flags |= MAP_HUGETLB;
                }
                auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, this->shm_fd, 0);
                if (ptr == MAP_FAILED) {
                    return nullptr;
                }
                return reinterpret_cast<uint8_t*>(ptr);
            }

            // 尝试加入已存在的共享内存
            JoinResult try_join_existing() {
                this->shm_fd = shm_open(this->storage_name.c_str(), O_RDWR, 0660);
                if (this->shm_fd == -1) {
                    if (errno == ENOENT) {
                        return JoinResult::FILE_NOT_FOUND;
                    }
                    log_msg("ERROR", "shm_open 失败: " + std::string(strerror(errno)));
                    return JoinResult::SYSTEM_ERROR;
                }

                // 预读 Header
                auto temp_ptr = map_memory_segment(MIN_MAP_SIZE, false);
                if (!temp_ptr) {
                    log_msg("ERROR", "读取 Header mmap 失败");
                    close(this->shm_fd);
                    return JoinResult::SYSTEM_ERROR;
                }

                auto header = reinterpret_cast<ShmMapHeader*>(temp_ptr);

                // 等待初始化完成 (Magic Check)
                int wait_count = 0;
                while (header->magic != MAGIC_CODE) {
                    wait_count += 1;
                    if (wait_count > 2000) {
                        log_msg("ERROR", "Magic 校验超时，文件可能已损坏");
                        munmap(temp_ptr, MIN_MAP_SIZE);
                        close(this->shm_fd);
                        shm_unlink(this->storage_name.c_str()); // 清理
                        return JoinResult::DATA_CORRUPT;
                    }
                    std::this_thread::sleep_for(milliseconds(1));
                    std::atomic_thread_fence(std::memory_order_acquire);
                }

                // 校验数据结构一致性
                if (header->align_param != ALIGN || header->value_size != sizeof(T) ||
                    header->padded_value_size != sizeof(PaddedValue<T, ALIGN>)) {

                    stringstream ss;
                    ss << "结构不匹配: File(Align=" << header->align_param << ", T=" << header->value_size << ") vs "
                       << "Code(Align=" << ALIGN << ", T=" << sizeof(T) << ")";
                    log_msg("ERROR", ss.str());

                    munmap(temp_ptr, MIN_MAP_SIZE);
                    close(this->shm_fd);
                    return JoinResult::TYPE_MISMATCH;
                }

                uint64_t full_size = header->total_file_size;
                munmap(temp_ptr, MIN_MAP_SIZE);

                // 完整映射
                this->mapped_ptr = map_memory_segment(full_size, true);
                if (!this->mapped_ptr) {
                    log_msg("WARN", "HugePage 映射失败，尝试降级");
                    this->mapped_ptr = map_memory_segment(full_size, false);
                    if (!this->mapped_ptr) {
                        close(this->shm_fd);
                        return JoinResult::SYSTEM_ERROR;
                    }
                }

                this->mapped_size = full_size;
                this->view.init(this->mapped_ptr);
                log_msg("INFO", "成功 Join 现有共享内存");
                return JoinResult::SUCCESS;
            }

            // // 内部结构：完美哈希构建结果
            // struct BuildResult {
            //         bool success = false;
            //         uint64_t bucket_mask = 0;
            //         uint64_t slot_mask = 0;
            //         uint32_t bucket_shift = 0;
            //         std::vector<uint64_t> control;
            //         std::vector<uint64_t> keys;
            //         std::vector<T> values;
            // };

            class BuildResult {
                public:
                    bool success = false;
                    uint64_t bucket_mask = 0;
                    uint64_t slot_mask = 0;
                    uint32_t bucket_shift = 0;
                    std::vector<uint64_t> control;
                    std::vector<uint64_t> keys;

                    std::vector<size_t> value_indices;
            };

            // 完美哈希核心算法 (与原代码保持一致)
            static uint64_t next_pow2(uint64_t x) {
                if (x == 0) {
                    return 1;
                }
                x -= 1;
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                x |= x >> 32;
                return x + 1;
            }

            BuildResult build_perfect_hash_tables(const std::vector<std::pair<uint64_t, T>>& data) {
                size_t n = data.size();
                if (n == 0)
                    return {};

                // 1. 安全检查 (O(N))
                for (const auto& kv : data) {
                    if (kv.first == EMPTY_KEY) {
                        throw std::runtime_error("Fatal: Input data contains reserved EMPTY_KEY (~0ULL)");
                    }
                }

                BuildResult res;
                double slot_factor = 1.1;
                double bucket_factor = 0.7;

                class BucketInfo {
                    public:
                        uint64_t id;
                        std::vector<size_t> data_indices;
                };

                while (true) {
                    uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
                    if (slot_cnt < 1024)
                        slot_cnt = 1024;

                    // 桶的数量不需要太大，减少 control table 大小
                    uint64_t bucket_cnt = next_pow2(static_cast<uint64_t>(n * bucket_factor));
                    if (bucket_cnt < 4)
                        bucket_cnt = 4;

                    res.bucket_mask = bucket_cnt - 1;
                    res.slot_mask = slot_cnt - 1;

                    int bucket_bits = 0;
                    while ((1ULL << bucket_bits) < bucket_cnt)
                        bucket_bits += 1;
                    res.bucket_shift = 64 - bucket_bits;

                    std::vector<BucketInfo> buckets(bucket_cnt);
                    for (uint64_t i = 0; i < bucket_cnt; i += 1)
                        buckets[i].id = i;

                    // [优化] 存下标，减少拷贝
                    for (size_t i = 0; i < n; i += 1) {
                        uint64_t h = HashCore::hash_one_pass(data[i].first);
                        uint64_t b_idx = h >> res.bucket_shift;
                        buckets[b_idx].data_indices.push_back(i);
                    }

                    // 排序：优先处理大桶（难处理的）
                    std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b) {
                        return a.data_indices.size() > b.data_indices.size();
                    });

                    res.control.assign(bucket_cnt, 0);
                    res.keys.assign(slot_cnt, EMPTY_KEY);
                    // [修改] 初始化为无效下标
                    res.value_indices.assign(slot_cnt, SIZE_MAX);

                    std::vector<bool> slot_used(slot_cnt, false);
                    std::mt19937_64 rng(123456); // 固定种子保证确定性
                    bool success = true;

                    for (const auto& bucket : buckets) {
                        if (bucket.data_indices.empty())
                            continue;

                        bool found_seed = false;
                        std::vector<uint64_t> proposed_slots;
                        proposed_slots.reserve(bucket.data_indices.size());

                        for (int attempt = 0; attempt < 2000000; attempt += 1) {
                            uint64_t seed = rng();
                            if (seed == 0)
                                seed = 1;

                            bool collision = false;
                            proposed_slots.clear();

                            for (size_t d_idx : bucket.data_indices) {
                                uint64_t k = data[d_idx].first; // [间接寻址]
                                uint64_t s_idx = (HashCore::hash_one_pass(k) ^ seed) & res.slot_mask;

                                if (slot_used[s_idx]) {
                                    collision = true;
                                    break;
                                }

                                // 检查桶内自冲突 (Self-collision)
                                for (auto ps : proposed_slots) {
                                    if (ps == s_idx) {
                                        collision = true;
                                        break;
                                    }
                                }

                                if (collision)
                                    break;
                                proposed_slots.push_back(s_idx);
                            }

                            if (!collision) {
                                res.control[bucket.id] = seed;
                                for (size_t i = 0; i < bucket.data_indices.size(); i += 1) {
                                    size_t original_idx = bucket.data_indices[i];
                                    uint64_t s_idx = proposed_slots[i];

                                    slot_used[s_idx] = true;
                                    res.keys[s_idx] = data[original_idx].first;
                                    res.value_indices[s_idx] = original_idx; // [关键] 记录原始下标
                                }
                                found_seed = true;
                                break;
                            }
                        }

                        if (!found_seed) {
                            success = false;
                            break;
                        }
                    }

                    if (success) {
                        res.success = true;
                        return res;
                    }

                    slot_factor *= 1.05; // 失败后扩容
                    bucket_factor *= 1.05;
                    if (slot_factor > 10.0)
                        return {};
                }
            }

            // BuildResult build_perfect_hash_tables(const std::vector<std::pair<uint64_t, T>>& data) {
            //     size_t n = data.size();
            //     if (n == 0) {
            //         return {};
            //     }

            //     for (const auto& kv : data) {
            //         if (kv.first == EMPTY_KEY) {
            //             throw std::runtime_error("Input data contains reserved EMPTY_KEY (~0ULL)");
            //         }
            //     }

            //     BuildResult res;
            //     double slot_factor = 1.1;
            //     double bucket_factor = 0.8;

            //     class BucketInfo {
            //         public:
            //             uint64_t id;
            //             std::vector<uint64_t> keys;
            //     };

            //     while (true) {
            //         uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
            //         if (slot_cnt < 1024) {
            //             slot_cnt = 1024;
            //         }
            //         uint64_t bucket_cnt = next_pow2(static_cast<uint64_t>(n * bucket_factor));
            //         if (bucket_cnt < 4) {
            //             bucket_cnt = 4;
            //         }

            //         res.bucket_mask = bucket_cnt - 1;
            //         res.slot_mask = slot_cnt - 1;

            //         int bucket_bits = 0;
            //         while ((1ULL << bucket_bits) < bucket_cnt)
            //             bucket_bits += 1;
            //         res.bucket_shift = 64 - bucket_bits;

            //         std::vector<BucketInfo> buckets(bucket_cnt);
            //         for (uint64_t i = 0; i < bucket_cnt; i += 1)
            //             buckets[i].id = i;

            //         for (const auto& kv : data) {
            //             uint64_t h = HashCore::hash_one_pass(kv.first);
            //             uint64_t b_idx = h >> res.bucket_shift;
            //             buckets[b_idx].keys.push_back(kv.first);
            //         }

            //         std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b) {
            //             return a.keys.size() > b.keys.size();
            //         });

            //         res.control.assign(bucket_cnt, 0);
            //         res.keys.assign(slot_cnt, EMPTY_KEY);
            //         res.values.resize(slot_cnt);

            //         std::vector<bool> slot_used(slot_cnt, false);
            //         std::mt19937_64 rng(123456);
            //         bool success = true;

            //         for (const auto& bucket : buckets) {
            //             if (bucket.keys.empty()) {
            //                 continue;
            //             }
            //             bool found_seed = false;
            //             std::vector<uint64_t> proposed_slots;
            //             proposed_slots.reserve(bucket.keys.size());

            //             for (int attempt = 0; attempt < 1000000; attempt += 1) {
            //                 uint64_t seed = rng();
            //                 if (seed == 0) {
            //                     seed = 1;
            //                 }
            //                 bool collision = false;
            //                 proposed_slots.clear();

            //                 for (uint64_t k : bucket.keys) {
            //                     uint64_t h = HashCore::hash_one_pass(k);
            //                     uint64_t s_idx = (h ^ seed) & res.slot_mask;
            //                     if (slot_used[s_idx]) {
            //                         collision = true;
            //                         break;
            //                     }
            //                     for (auto ps : proposed_slots)
            //                         if (ps == s_idx) {
            //                             collision = true;
            //                             break;
            //                         }
            //                     if (collision) {
            //                         break;
            //                     }
            //                     proposed_slots.push_back(s_idx);
            //                 }

            //                 if (!collision) {
            //                     res.control[bucket.id] = seed;
            //                     for (size_t i = 0; i < bucket.keys.size(); i += 1) {
            //                         uint64_t s_idx = proposed_slots[i];
            //                         slot_used[s_idx] = true;
            //                         res.keys[s_idx] = bucket.keys[i];
            //                         // 查找原始 Value
            //                         for (const auto& d : data)
            //                             if (d.first == bucket.keys[i]) {
            //                                 res.values[s_idx] = d.second;
            //                                 break;
            //                             }
            //                     }
            //                     found_seed = true;
            //                     break;
            //                 }
            //             }
            //             if (!found_seed) {
            //                 success = false;
            //                 break;
            //             }
            //         }

            //         if (success) {
            //             res.success = true;
            //             return res;
            //         }
            //         slot_factor *= 1.1;
            //         bucket_factor *= 1.1;
            //         if (slot_factor > 20.0) {
            //             return {};
            //         }
            //     }
            // }

            bool try_create_new(const std::vector<std::pair<uint64_t, T>>& data) {
                if (data.empty()) {
                    log_msg("ERROR", "无法创建新 SHM: 初始化数据为空");
                    return false;
                }

                // 构建内存中的完美哈希表
                auto build_res = build_perfect_hash_tables(data);
                if (!build_res.success) {
                    log_msg("ERROR", "完美哈希构建失败 (数据分布过于离散或冲突严重)");
                    return false;
                }

                // 计算内存布局
                size_t header_sz = align_to_cache_line(sizeof(ShmMapHeader));
                size_t ctrl_sz = align_to_cache_line(build_res.control.size() * sizeof(uint64_t));
                size_t key_sz = align_to_cache_line(build_res.keys.size() * sizeof(uint64_t));
                size_t val_sz = align_to_cache_line(build_res.value_indices.size() * sizeof(PaddedValue<T, ALIGN>));

                size_t total_sz = header_sz + ctrl_sz + key_sz + val_sz;
                size_t aligned_file_sz = align_to_huge_page(total_sz);
                size_t waste_sz = aligned_file_sz - total_sz;

                {
                    stringstream ss;
                    ss << "开始创建共享内存 (Perfect Hash Map):\n"
                       << "  名称            = " << this->storage_name << "\n"
                       << "  逻辑数据量      = " << data.size() << " (实际写入)\n"
                       << "  物理槽位数量    = " << build_res.keys.size() << " (Slot Count, Load Factor ≈ "
                       << std::fixed << std::setprecision(2) << ((double) data.size() / build_res.keys.size()) << ")\n"
                       << "  sizeof(T)       = " << sizeof(T) << " 字节\n"
                       << "  sizeof(Padded)  = " << sizeof(PaddedValue<T, ALIGN>) << " 字节\n"
                       << "  alignof(T)      = " << alignof(T) << "\n"
                       << "  alignof(Padded) = " << alignof(PaddedValue<T, ALIGN>) << " (CacheLine=" << ALIGN << ")";
                    log_msg("INFO", ss.str());
                }

                {
                    stringstream ss;
                    ss << "内存布局计算详情:\n"
                       << "  [1] Header 区     = " << header_sz << " 字节\n"
                       << "  [2] Control 表    = " << ctrl_sz << " 字节\n"
                       << "  [3] Key 表        = " << key_sz << " 字节\n"
                       << "  [4] Value 表      = " << val_sz << " 字节\n"
                       << "  --------------------------------\n"
                       << "  实际数据总大小    = " << total_sz << " 字节\n"
                       << "  HugePage对齐后    = " << aligned_file_sz << " 字节\n"
                       << "  对齐浪费空间      = " << waste_sz << " 字节 (" << std::fixed << std::setprecision(2)
                       << (100.0 * waste_sz / aligned_file_sz) << "%)";
                    log_msg("INFO", ss.str());
                }

                // 创建文件 (O_EXCL)
                this->shm_fd = shm_open(this->storage_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0660);
                if (this->shm_fd == -1) {
                    return false; // 可能被别人抢先创建了
                }

                if (ftruncate(this->shm_fd, aligned_file_sz) == -1) {
                    log_msg("ERROR", "ftruncate 失败");
                    close(this->shm_fd);
                    shm_unlink(this->storage_name.c_str());
                    return false;
                }

                // 映射
                this->mapped_ptr = map_memory_segment(aligned_file_sz, true);
                if (!this->mapped_ptr) {
                    log_msg("WARN", "HugePage 失败，降级");
                    this->mapped_ptr = map_memory_segment(aligned_file_sz, false);
                    if (!this->mapped_ptr) {
                        close(this->shm_fd);
                        shm_unlink(this->storage_name.c_str());
                        return false;
                    }
                }
                this->mapped_size = aligned_file_sz;

                // 写入数据
                auto* hdr = new (this->mapped_ptr) ShmMapHeader();
                hdr->bucket_mask = build_res.bucket_mask;
                hdr->slot_mask = build_res.slot_mask;
                hdr->bucket_shift = build_res.bucket_shift;
                hdr->item_count = data.size();
                hdr->bucket_count = build_res.control.size();
                hdr->slot_count = build_res.keys.size();

                hdr->total_file_size = aligned_file_sz;
                hdr->align_param = ALIGN;
                hdr->value_size = sizeof(T);
                hdr->padded_value_size = sizeof(PaddedValue<T, ALIGN>);

                hdr->offset_control = header_sz;
                hdr->offset_keys = header_sz + ctrl_sz;
                hdr->offset_values = header_sz + ctrl_sz + key_sz;

                // 复制 Control Table
                std::memcpy(this->mapped_ptr + hdr->offset_control, build_res.control.data(),
                            build_res.control.size() * sizeof(uint64_t));

                // 复制 Key Table
                std::memcpy(this->mapped_ptr + hdr->offset_keys, build_res.keys.data(),
                            build_res.keys.size() * sizeof(uint64_t));

                auto* val_ptr = reinterpret_cast<PaddedValue<T, ALIGN>*>(this->mapped_ptr + hdr->offset_values);
                // 注意：遍历的是 slot (0 到 slot_count)
                for (size_t i = 0; i < build_res.value_indices.size(); i += 1) {
                    size_t src_idx = build_res.value_indices[i];

                    // 构造对象
                    new (&val_ptr[i]) PaddedValue<T, ALIGN>();

                    if (src_idx != SIZE_MAX) {
                        val_ptr[i].value = data[src_idx].second;
                    }
                    val_ptr[i].unlock();
                }

                // 发布 (Memory Barrier & Magic)
                std::atomic_thread_fence(std::memory_order_release);
                hdr->magic = MAGIC_CODE;

                this->view.init(this->mapped_ptr);
                {
                    stringstream ss;
                    ss << "共享内存创建成功并已发布:\n"
                       << "  映射总大小        = " << aligned_file_sz << " 字节 (" 
                       << std::fixed << std::setprecision(2) << (aligned_file_sz / 1024.0 / 1024.0) << " MB)";
                    log_msg("INFO", ss.str());
                }
                return true;
            }

        public:
            // 返回 bool: true = 只是 Join (只读或更新), false = 刚刚 Create (新建)
            // 如果 init_data 为空且文件不存在，将抛出异常
            bool build(std::string name, const std::vector<std::pair<uint64_t, T>>& init_data = {}) {
                if (geteuid() != 0) {
                    throw std::runtime_error("需要 Root 权限 (HugePage)");
                }
                this->storage_name = name;

                for (auto i = 0; i < 3; i += 1) {
                    // 尝试 Join
                    auto join_res = try_join_existing();
                    if (join_res == JoinResult::SUCCESS) {
                        return true;
                    }

                    if (join_res == JoinResult::TYPE_MISMATCH) {
                        throw std::runtime_error("共享内存数据结构版本不匹配");
                    }

                    if (join_res == JoinResult::DATA_CORRUPT) {
                        log_msg("WARN", "检测到损坏的文件，已删除并重试创建...");
                        continue;
                    }

                    // 尝试 Create (FILE_NOT_FOUND)
                    if (try_create_new(init_data)) {
                        return false;
                    }

                    // 处理并发竞争 (EEXIST)
                    if (errno == EEXIST) {
                        log_msg("WARN", "检测到并发竞争 (EEXIST)，正在重试 join...");
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        continue;
                    }

                    throw std::runtime_error("shm_open 致命错误: " + string(strerror(errno)));
                }
                throw std::runtime_error("由于严重的并发竞争，初始化超时");
            }

            inline SharedMapView<T, ALIGN>& get_view() {
                return this->view;
            }

            ~ShmMapStorage() {
                if (this->mapped_ptr) {
                    munmap(this->mapped_ptr, this->mapped_size);
                }
                if (shm_fd != -1) {
                    close(shm_fd);
                }
            }
    };
} // namespace shm_pm

#pragma pack(pop)
#endif //HASH_STORAGE_HPP