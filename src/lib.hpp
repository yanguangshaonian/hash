// #ifndef HASH_HPP
// #define HASH_HPP
// #pragma pack(push)
// #pragma pack()


// #include <iostream>
// #include <vector>
// #include <string>
// #include <atomic>
// #include <algorithm>
// #include <random>
// #include <cstring>
// #include <cstdint>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <immintrin.h>
// #include <thread>
// #include <iomanip>
// #include <stdexcept>
// #include <sstream>

// namespace shm_pm {

//     using namespace std;
//     using namespace std::chrono;

//     // ----------------------------------------------------------------
//     // 基础常量与工具
//     // ----------------------------------------------------------------
//     constexpr size_t CACHE_LINE_SIZE = 64;
//     constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
//     constexpr uint64_t MAGIC_CODE = 0xDEADBEEFCAFEBABE;
//     constexpr uint64_t EMPTY_KEY = ~0ULL;

//     inline uint64_t align_to(uint64_t size, uint64_t alignment) {
//         return (size + alignment - 1) & ~(alignment - 1);
//     }

//     namespace HashCore {
//         // 高性能混合哈希 (Murmur/WyHash 变体)
//         __attribute__((always_inline)) inline uint64_t hash_one_pass(uint64_t k) {
//             k ^= k >> 33;
//             k *= 0xff51afd7ed558ccdULL;
//             k ^= k >> 33;
//             k *= 0xc4ceb9fe1a85ec53ULL;
//             k ^= k >> 33;
//             return k;
//         }
//     } // namespace HashCore

//     // ----------------------------------------------------------------
//     // 并发安全的数据单元 (自旋锁保护)
//     // ----------------------------------------------------------------
//     template<class T, size_t ALIGN = CACHE_LINE_SIZE>
//     class alignas(ALIGN) PaddedValue {
//         public:
//             T value;
//             std::atomic<bool> busy_flag = false;

//             // 指数退避之后 强制抢占锁
//             // allow_fail == true: 达到最大退避后允许失败返回
//             // allow_fail == false: 无限自旋直到获取锁
//             inline __attribute__((always_inline)) bool lock(bool allow_fail = true) noexcept {
//                 while (true) {
//                     uint8_t delay = 0b0000'0001;
//                     while (busy_flag.load(std::memory_order_relaxed)) {
//                         for (auto i = 0; i < delay; i += 1) {
//                             asm volatile("pause" ::: "memory");
//                         }

//                         if (delay < 0b1000'0000) {
//                             delay <<= 1;
//                         } else {
//                             // 已达到最大退避
//                             if (allow_fail) {
//                                 return false;
//                             }
//                         }
//                     }

//                     if (!busy_flag.exchange(true, std::memory_order_acquire)) {
//                         return true;
//                     }
//                 }
//             }

//             inline __attribute__((always_inline)) void unlock() noexcept {
//                 busy_flag.store(false, std::memory_order_release);
//             }
//     };

//     // ----------------------------------------------------------------
//     // 共享内存头部 (Header)
//     // ----------------------------------------------------------------
//     class alignas(CACHE_LINE_SIZE) ShmMapHeader {
//         public:
//             volatile uint64_t magic;

//             // 哈希参数
//             uint64_t bucket_mask; // hash >> bucket_shift 得到 bucket index
//             uint64_t slot_mask;   // (hash ^ seed) & slot_mask 得到 slot index
//             uint32_t bucket_shift;

//             // 统计信息
//             uint64_t item_count;   // 实际插入的元素数量
//             uint64_t bucket_count; // 桶数量
//             uint64_t slot_count;   // 槽位数量 (包含空闲)

//             // 偏移量 (相对于文件头)
//             uint64_t offset_control; // Seed 数组
//             uint64_t offset_keys;    // Key 数组
//             uint64_t offset_values;  // PaddedValue 数组 (可变区域)

//             // 文件总大小 (对齐后)
//             uint64_t total_file_size;
//     };

//     // ----------------------------------------------------------------
//     // 映射视图 (View) - 提供查找与访问能力
//     // ----------------------------------------------------------------
//     template<class T, size_t ALIGN = CACHE_LINE_SIZE>
//     class SharedMapView {
//         private:
//             const uint8_t* base_ptr = nullptr;
//             const ShmMapHeader* header = nullptr;

//             // 指针缓存
//             const uint64_t* control_table = nullptr; // Seeds
//             const uint64_t* key_table = nullptr;     // Keys
//             PaddedValue<T, ALIGN>* value_table = nullptr;   // Values (Mutable)

//         public:
//             void init(uint8_t* mapped_address) {
//                 base_ptr = mapped_address;
//                 header = reinterpret_cast<const ShmMapHeader*>(base_ptr);

//                 if (header->magic != MAGIC_CODE) {
//                     throw std::runtime_error("SharedMapView: Magic number mismatch.");
//                 }

//                 control_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_control);
//                 key_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_keys);
//                 value_table = reinterpret_cast<PaddedValue<T, ALIGN>*>(mapped_address + header->offset_values);
//             }

//             __attribute__((always_inline)) const T* get(uint64_t key) {
//                 uint64_t h = HashCore::hash_one_pass(key);
//                 uint64_t bucket_idx = h >> header->bucket_shift;
//                 uint64_t seed = control_table[bucket_idx];
//                 uint64_t slot_idx = (h ^ seed) & header->slot_mask;

//                 uint64_t stored_key = key_table[slot_idx];
//                 if (__builtin_expect(stored_key == key, 1)) {
//                     return &value_table[slot_idx].value;
//                 }
//                 return nullptr;
//             }

//             // 获取统计信息
//             size_t capacity() const {
//                 return header ? header->slot_count : 0;
//             }

//             size_t size() const {
//                 return header ? header->item_count : 0;
//             }
//     };

//     // ----------------------------------------------------------------
//     // 存储管理器
//     // ----------------------------------------------------------------
//     template<class T, size_t ALIGN = CACHE_LINE_SIZE>
//     class ShmMapStorage {
//         private:
//             int shm_fd = -1;
//             uint8_t* mapped_ptr = nullptr;
//             size_t mapped_size = 0;
//             std::string name;
//             SharedMapView<T, ALIGN> view;

//             // 辅助: 下一个 2 的幂
//             static uint64_t next_pow2(uint64_t x) {
//                 if (x == 0)
//                     return 1;
//                 x--;
//                 x |= x >> 1;
//                 x |= x >> 2;
//                 x |= x >> 4;
//                 x |= x >> 8;
//                 x |= x >> 16;
//                 x |= x >> 32;
//                 return x + 1;
//             }

//             // 辅助: 日志
//             void log(const std::string& level, const std::string& msg) {
//                 std::cout << "[ShmMap][" << level << "] " << msg << std::endl;
//             }

//         public:
//             ShmMapStorage() = default;

//             ~ShmMapStorage() {
//                 close_shm();
//             }

//             void close_shm() {
//                 if (mapped_ptr) {
//                     munmap(mapped_ptr, mapped_size);
//                     mapped_ptr = nullptr;
//                 }
//                 if (shm_fd != -1) {
//                     close(shm_fd);
//                     shm_fd = -1;
//                 }
//             }

//             SharedMapView<T, ALIGN>& get_view() {
//                 return view;
//             }

//             // 连接到已存在的 SHM
//             bool open_existing(const std::string& storage_name) {
//                 this->name = storage_name;
//                 shm_fd = shm_open(name.c_str(), O_RDWR, 0660);
//                 if (shm_fd == -1)
//                     return false;

//                 // 1. 读取头部以获取大小
//                 void* hdr_ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
//                 if (hdr_ptr == MAP_FAILED) {
//                     close(shm_fd);
//                     return false;
//                 }

//                 auto* temp_header = reinterpret_cast<ShmMapHeader*>(hdr_ptr);
//                 // 等待 Magic (简单的自旋等待初始化完成)
//                 int retries = 0;
//                 while (temp_header->magic != MAGIC_CODE && retries < 1000) {
//                     std::this_thread::sleep_for(std::chrono::milliseconds(1));
//                     retries += 1;
//                 }

//                 if (temp_header->magic != MAGIC_CODE) {
//                     munmap(hdr_ptr, 4096);
//                     close(shm_fd);
//                     log("ERROR", "Shared memory exists but magic is invalid or not ready.");
//                     return false;
//                 }

//                 size_t full_size = temp_header->total_file_size;
//                 munmap(hdr_ptr, 4096);

//                 // 完整映射 (尝试 HugePage)
//                 void* ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, shm_fd, 0);
//                 if (ptr == MAP_FAILED) {
//                     // 降级到普通页
//                     ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
//                     if (ptr == MAP_FAILED) {
//                         close(shm_fd);
//                         log("ERROR", "Failed to mmap full file.");
//                         return false;
//                     }
//                 }

//                 mapped_ptr = static_cast<uint8_t*>(ptr);
//                 mapped_size = full_size;

//                 // 初始化视图
//                 view.init(mapped_ptr);
//                 log("INFO", "Opened existing SHM successfully.");
//                 return true;
//             }

//             // 创建新的 SHM (需要源数据)
//             // data: 原始键值对
//             bool create_new(const std::string& storage_name, const std::vector<std::pair<uint64_t, T>>& data) {
//                 this->name = storage_name;

//                 // 计算完美哈希 (在内存中进行, 不占用 SHM 锁)
//                 auto build_res = build_perfect_hash_tables(data);
//                 if (build_res.success == false) {
//                     log("ERROR", "Failed to build perfect hash tables.");
//                     return false;
//                 }

//                 // 计算布局与大小
//                 size_t header_sz = align_to(sizeof(ShmMapHeader), CACHE_LINE_SIZE);
//                 size_t ctrl_sz = align_to(build_res.control.size() * sizeof(uint64_t), CACHE_LINE_SIZE);
//                 size_t key_sz = align_to(build_res.keys.size() * sizeof(uint64_t), CACHE_LINE_SIZE);
//                 size_t val_sz = align_to(build_res.values.size() * sizeof(PaddedValue<T, ALIGN>), CACHE_LINE_SIZE);

//                 size_t total_sz = header_sz + ctrl_sz + key_sz + val_sz;
//                 size_t aligned_file_sz = align_to(total_sz, HUGE_PAGE_SIZE); // 对齐到 2MB

//                 // 3. 创建 SHM 文件 (O_EXCL 保证我是创建者)
//                 shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0660);
//                 if (shm_fd == -1) {
//                     log("WARN", "shm_open failed (O_EXCL), file might exist.");
//                     return false;
//                 }

//                 if (ftruncate(shm_fd, aligned_file_sz) == -1) {
//                     log("ERROR", "ftruncate failed.");
//                     shm_unlink(name.c_str());
//                     close(shm_fd);
//                     return false;
//                 }

//                 // 4. 映射
//                 void* ptr = mmap(nullptr, aligned_file_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, shm_fd, 0);
//                 if (ptr == MAP_FAILED) {
//                     ptr = mmap(nullptr, aligned_file_sz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
//                     if (ptr == MAP_FAILED) {
//                         log("ERROR", "mmap failed during creation.");
//                         shm_unlink(name.c_str());
//                         close(shm_fd);
//                         return false;
//                     }
//                 }
//                 mapped_ptr = static_cast<uint8_t*>(ptr);
//                 mapped_size = aligned_file_sz;

//                 // 填充数据
//                 // 填充 Header
//                 auto* hdr = new (mapped_ptr) ShmMapHeader();
//                 hdr->bucket_mask = build_res.bucket_mask;
//                 hdr->slot_mask = build_res.slot_mask;
//                 hdr->bucket_shift = build_res.bucket_shift;
//                 hdr->item_count = data.size();
//                 hdr->bucket_count = build_res.control.size();
//                 hdr->slot_count = build_res.keys.size();
//                 hdr->total_file_size = aligned_file_sz;

//                 hdr->offset_control = header_sz;
//                 hdr->offset_keys = header_sz + ctrl_sz;
//                 hdr->offset_values = header_sz + ctrl_sz + key_sz;

//                 // 填充 Tables
//                 std::memcpy(mapped_ptr + hdr->offset_control, build_res.control.data(),
//                             build_res.control.size() * sizeof(uint64_t));
//                 std::memcpy(mapped_ptr + hdr->offset_keys, build_res.keys.data(),
//                             build_res.keys.size() * sizeof(uint64_t));

//                 // 填充 Padded Values (需要逐个构造)
//                 auto* val_ptr = reinterpret_cast<PaddedValue<T, ALIGN>*>(mapped_ptr + hdr->offset_values);
//                 for (size_t i = 0; i < build_res.values.size(); i += 1) {
//                     // 使用 placement new 初始化 atomic
//                     new (&val_ptr[i]) PaddedValue<T, ALIGN>();
//                     val_ptr[i].value = build_res.values[i];
//                     val_ptr[i].unlock(); // 确保存储为 false
//                 }

//                 // 提交 Magic (Memory Barrier)
//                 std::atomic_thread_fence(std::memory_order_release);
//                 hdr->magic = MAGIC_CODE;

//                 // 初始化视图
//                 view.init(mapped_ptr);
//                 log("INFO", "Created new SHM Map successfully.");
//                 return true;
//             }

//         private:
//             // 构建结果容器
//             class BuildResult {
//                 public:
//                     bool success = false;
//                     uint64_t bucket_mask = 0;
//                     uint64_t slot_mask = 0;
//                     uint32_t bucket_shift = 0;
//                     std::vector<uint64_t> control;
//                     std::vector<uint64_t> keys;
//                     std::vector<T> values;
//             };

//             // 核心算法: 生成 Perfect Hash Tables (CPU 密集型)
//             BuildResult build_perfect_hash_tables(const std::vector<std::pair<uint64_t, T>>& data) {
//                 size_t n = data.size();
//                 if (n == 0) {
//                     return {};
//                 }

//                 BuildResult res;
//                 double slot_factor = 2.0;
//                 double bucket_factor = 0.8;

//                 // 内部结构
//                 class BucketInfo {
//                     public:
//                         uint64_t id;
//                         std::vector<uint64_t> keys;
//                 };

//                 while (true) {
//                     uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
//                     if (slot_cnt < 1024) {
//                         slot_cnt = 1024;
//                     }
//                     uint64_t bucket_cnt = next_pow2(static_cast<uint64_t>(n * bucket_factor));
//                     if (bucket_cnt < 4) {
//                         bucket_cnt = 4;
//                     }

//                     // 准备 Bucket
//                     res.bucket_mask = bucket_cnt - 1;
//                     res.slot_mask = slot_cnt - 1;
//                     int bucket_bits = 0;
//                     while ((1ULL << bucket_bits) < bucket_cnt) {
//                         bucket_bits += 1;
//                     }
//                     res.bucket_shift = 64 - bucket_bits;

//                     std::vector<BucketInfo> buckets(bucket_cnt);
//                     for (uint64_t i = 0; i < bucket_cnt; i += 1) {
//                         buckets[i].id = i;
//                     }

//                     for (const auto& kv : data) {
//                         if (kv.first == EMPTY_KEY) {
//                             std::cerr << "[Fatal] Key cannot be ~0ULL (Reserved for EMPTY)" << std::endl;
//                             return {};
//                         }
//                         uint64_t h = HashCore::hash_one_pass(kv.first);
//                         uint64_t b_idx = h >> res.bucket_shift;
//                         buckets[b_idx].keys.push_back(kv.first);
//                     }

//                     // 排序 (Desc)
//                     std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b) {
//                         return a.keys.size() > b.keys.size();
//                     });

//                     // 寻找 Seeds
//                     res.control.assign(bucket_cnt, 0);
//                     res.keys.assign(slot_cnt, EMPTY_KEY);
//                     res.values.resize(slot_cnt);

//                     std::vector<bool> slot_used(slot_cnt, false);
//                     std::mt19937_64 rng(123456);
//                     bool success = true;

//                     for (const auto& bucket : buckets) {
//                         if (bucket.keys.empty())
//                             continue;

//                         bool found_seed = false;
//                         std::vector<uint64_t> proposed_slots;
//                         proposed_slots.reserve(bucket.keys.size());

//                         for (int attempt = 0; attempt < 1000000; attempt += 1) {
//                             uint64_t seed = rng();
//                             if (seed == 0) {
//                                 seed = 1;
//                             }

//                             bool collision = false;
//                             proposed_slots.clear();

//                             for (uint64_t k : bucket.keys) {
//                                 uint64_t h = HashCore::hash_one_pass(k);
//                                 uint64_t s_idx = (h ^ seed) & res.slot_mask;

//                                 if (slot_used[s_idx]) {
//                                     collision = true;
//                                     break;
//                                 }
//                                 for (auto ps : proposed_slots)
//                                     if (ps == s_idx) {
//                                         collision = true;
//                                         break;
//                                     }
//                                 if (collision)
//                                     break;
//                                 proposed_slots.push_back(s_idx);
//                             }

//                             if (!collision) {
//                                 res.control[bucket.id] = seed;
//                                 for (size_t i = 0; i < bucket.keys.size(); i += 1) {
//                                     uint64_t k = bucket.keys[i];
//                                     uint64_t s_idx = proposed_slots[i];
//                                     slot_used[s_idx] = true;
//                                     res.keys[s_idx] = k;

//                                     // 查找 Value (简单线性查找, 构建时性能非瓶颈)
//                                     for (const auto& input_kv : data) {
//                                         if (input_kv.first == k) {
//                                             res.values[s_idx] = input_kv.second;
//                                             break;
//                                         }
//                                     }
//                                 }
//                                 found_seed = true;
//                                 break;
//                             }
//                         }

//                         if (!found_seed) {
//                             success = false;
//                             break;
//                         }
//                     }

//                     if (success) {
//                         res.success = true;
//                         return res;
//                     }

//                     // 扩容重试
//                     slot_factor *= 1.3;
//                     bucket_factor *= 1.1;
//                     if (slot_factor > 10.0) {
//                         return {};
//                     }
//                 }
//             }
//     };
// } // namespace shm_pm

// #pragma pack(pop)
// #endif //HASH_HPP


#ifndef HASH_HPP
#define HASH_HPP
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

namespace shm_pm {
    using namespace std;
    using namespace std::chrono;

    // ----------------------------------------------------------------
    // 基础常量与工具
    // ----------------------------------------------------------------
    constexpr size_t CACHE_LINE_SIZE = 64;
    constexpr size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    constexpr uint64_t MAGIC_CODE = 0xDEADBEEF20260130;
    constexpr uint64_t EMPTY_KEY = ~0ULL;

    inline uint64_t align_to(uint64_t size, uint64_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
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
            // 确保 busy_flag 在 value 之后，且整体结构对齐
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
                            // 已达到最大退避
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
            volatile uint64_t magic;

            uint64_t bucket_mask;
            uint64_t slot_mask;
            uint32_t bucket_shift;

            uint64_t item_count;
            uint64_t bucket_count;
            uint64_t slot_count;

            uint64_t offset_control;
            uint64_t offset_keys;
            uint64_t offset_values;

            uint64_t total_file_size;
            uint64_t align_param;
    };

    // ----------------------------------------------------------------
    // 映射视图 (View)
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

                if (header->magic != MAGIC_CODE) {
                    throw std::runtime_error("SharedMapView: Magic number mismatch.");
                }
                // 安全检查: 确保读取端和写入端的对齐参数一致
                if (header->align_param != ALIGN) {
                    throw std::runtime_error("SharedMapView: ALIGN parameter mismatch with storage file.");
                }

                control_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_control);
                key_table = reinterpret_cast<const uint64_t*>(base_ptr + header->offset_keys);
                value_table = reinterpret_cast<PaddedValue<T, ALIGN>*>(mapped_address + header->offset_values);
            }

            // [只读] 极速获取，无锁，返回 const T*
            __attribute__((always_inline)) const T* get(uint64_t key) const {
                // 移除 EMPTY_KEY 检查以追求极致性能 (调用者需保证 key != EMPTY_KEY)
                uint64_t h = HashCore::hash_one_pass(key);
                uint64_t bucket_idx = h >> header->bucket_shift;
                uint64_t seed = control_table[bucket_idx];
                uint64_t slot_idx = (h ^ seed) & header->slot_mask;

                if (__builtin_expect(key_table[slot_idx] == key, 1)) {
                    return &value_table[slot_idx].value;
                }
                return nullptr;
            }

            // [新增] 获取 PaddedValue 指针，以便用户手动加锁
            __attribute__((always_inline)) PaddedValue<T, ALIGN>* get_entry(uint64_t key) {
                uint64_t h = HashCore::hash_one_pass(key);
                uint64_t bucket_idx = h >> header->bucket_shift;
                uint64_t seed = control_table[bucket_idx];
                uint64_t slot_idx = (h ^ seed) & header->slot_mask;

                if (__builtin_expect(key_table[slot_idx] == key, 1)) {
                    return &value_table[slot_idx];
                }
                return nullptr;
            }

            // [新增] 带锁操作的便捷函数
            template<typename Func>
            __attribute__((always_inline)) bool locked_access(uint64_t key, Func&& f) {
                auto* entry = get_entry(key);
                if (!entry)
                    return false;

                entry->lock(false); // false = 无限自旋等待
                f(entry->value);
                entry->unlock();
                return true;
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
            int shm_fd = -1;
            uint8_t* mapped_ptr = nullptr;
            size_t mapped_size = 0;
            std::string name;
            SharedMapView<T, ALIGN> view;

            static uint64_t next_pow2(uint64_t x) {
                if (x == 0)
                    return 1;
                x--;
                x |= x >> 1;
                x |= x >> 2;
                x |= x >> 4;
                x |= x >> 8;
                x |= x >> 16;
                x |= x >> 32;
                return x + 1;
            }

            void log(const std::string& level, const std::string& msg) {
                std::cout << "[ShmMap][" << level << "] " << msg << std::endl;
            }

        public:
            ShmMapStorage() = default;

            ~ShmMapStorage() {
                close_shm();
            }

            void close_shm() {
                if (mapped_ptr) {
                    munmap(mapped_ptr, mapped_size);
                    mapped_ptr = nullptr;
                }
                if (shm_fd != -1) {
                    close(shm_fd);
                    shm_fd = -1;
                }
            }

            SharedMapView<T, ALIGN>& get_view() {
                return view;
            }

            bool open_existing(const std::string& storage_name) {
                this->name = storage_name;
                shm_fd = shm_open(name.c_str(), O_RDWR, 0660);
                if (shm_fd == -1)
                    return false;

                void* hdr_ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                if (hdr_ptr == MAP_FAILED) {
                    close(shm_fd);
                    return false;
                }

                auto* temp_header = reinterpret_cast<ShmMapHeader*>(hdr_ptr);
                int retries = 0;
                while (temp_header->magic != MAGIC_CODE && retries < 1000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    retries += 1;
                }

                if (temp_header->magic != MAGIC_CODE) {
                    munmap(hdr_ptr, 4096);
                    close(shm_fd);
                    return false;
                }

                size_t full_size = temp_header->total_file_size;
                munmap(hdr_ptr, 4096);

                void* ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, shm_fd, 0);
                if (ptr == MAP_FAILED) {
                    ptr = mmap(nullptr, full_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                    if (ptr == MAP_FAILED) {
                        close(shm_fd);
                        return false;
                    }
                }

                mapped_ptr = static_cast<uint8_t*>(ptr);
                mapped_size = full_size;
                view.init(mapped_ptr); // 这里会检查 ALIGN 是否匹配
                log("INFO", "Opened existing SHM successfully.");
                return true;
            }

            bool create_new(const std::string& storage_name, const std::vector<std::pair<uint64_t, T>>& data) {
                this->name = storage_name;

                auto build_res = build_perfect_hash_tables(data);
                if (!build_res.success)
                    return false;

                size_t header_sz = align_to(sizeof(ShmMapHeader), CACHE_LINE_SIZE);
                size_t ctrl_sz = align_to(build_res.control.size() * sizeof(uint64_t), CACHE_LINE_SIZE);
                size_t key_sz = align_to(build_res.keys.size() * sizeof(uint64_t), CACHE_LINE_SIZE);
                // 正确使用了 ALIGN 模板参数
                size_t val_sz = align_to(build_res.values.size() * sizeof(PaddedValue<T, ALIGN>), CACHE_LINE_SIZE);

                size_t total_sz = header_sz + ctrl_sz + key_sz + val_sz;
                size_t aligned_file_sz = align_to(total_sz, HUGE_PAGE_SIZE);

                shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0660);
                if (shm_fd == -1)
                    return false;

                if (ftruncate(shm_fd, aligned_file_sz) == -1) {
                    close(shm_fd);
                    shm_unlink(name.c_str());
                    return false;
                }

                void* ptr = mmap(nullptr, aligned_file_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, shm_fd, 0);
                if (ptr == MAP_FAILED) {
                    ptr = mmap(nullptr, aligned_file_sz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                    if (ptr == MAP_FAILED) {
                        close(shm_fd);
                        shm_unlink(name.c_str());
                        return false;
                    }
                }
                mapped_ptr = static_cast<uint8_t*>(ptr);
                mapped_size = aligned_file_sz;

                auto* hdr = new (mapped_ptr) ShmMapHeader();
                hdr->bucket_mask = build_res.bucket_mask;
                hdr->slot_mask = build_res.slot_mask;
                hdr->bucket_shift = build_res.bucket_shift;
                hdr->item_count = data.size();
                hdr->bucket_count = build_res.control.size();
                hdr->slot_count = build_res.keys.size();
                hdr->total_file_size = aligned_file_sz;
                // 记录 ALIGN，方便 open 时校验
                hdr->align_param = ALIGN;

                hdr->offset_control = header_sz;
                hdr->offset_keys = header_sz + ctrl_sz;
                hdr->offset_values = header_sz + ctrl_sz + key_sz;

                std::memcpy(mapped_ptr + hdr->offset_control, build_res.control.data(),
                            build_res.control.size() * sizeof(uint64_t));
                std::memcpy(mapped_ptr + hdr->offset_keys, build_res.keys.data(),
                            build_res.keys.size() * sizeof(uint64_t));

                auto* val_ptr = reinterpret_cast<PaddedValue<T, ALIGN>*>(mapped_ptr + hdr->offset_values);
                for (size_t i = 0; i < build_res.values.size(); i += 1) {
                    new (&val_ptr[i]) PaddedValue<T, ALIGN>();
                    val_ptr[i].value = build_res.values[i];
                    val_ptr[i].unlock();
                }

                std::atomic_thread_fence(std::memory_order_release);
                hdr->magic = MAGIC_CODE;

                view.init(mapped_ptr);
                log("INFO", "Created new SHM Map successfully.");
                return true;
            }

        private:
            // ... (BuildResult 和 build_perfect_hash_tables 保持不变)
            struct BuildResult {
                    bool success = false;
                    uint64_t bucket_mask = 0;
                    uint64_t slot_mask = 0;
                    uint32_t bucket_shift = 0;
                    std::vector<uint64_t> control;
                    std::vector<uint64_t> keys;
                    std::vector<T> values;
            };

            BuildResult build_perfect_hash_tables(const std::vector<std::pair<uint64_t, T>>& data) {
                size_t n = data.size();
                if (n == 0)
                    return {};

                BuildResult res;
                double slot_factor = 2.0;
                double bucket_factor = 0.8;

                class BucketInfo {
                    public:
                        uint64_t id;
                        std::vector<uint64_t> keys;
                };

                while (true) {
                    uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
                    if (slot_cnt < 1024)
                        slot_cnt = 1024;
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

                    for (const auto& kv : data) {
                        if (kv.first == EMPTY_KEY) {
                            std::cerr << "[Fatal] Key cannot be ~0ULL (Reserved for EMPTY)" << std::endl;
                            return {};
                        }
                        uint64_t h = HashCore::hash_one_pass(kv.first);
                        uint64_t b_idx = h >> res.bucket_shift;
                        buckets[b_idx].keys.push_back(kv.first);
                    }

                    std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b) {
                        return a.keys.size() > b.keys.size();
                    });

                    res.control.assign(bucket_cnt, 0);
                    res.keys.assign(slot_cnt, EMPTY_KEY);
                    res.values.resize(slot_cnt);

                    std::vector<bool> slot_used(slot_cnt, false);
                    std::mt19937_64 rng(123456);
                    bool success = true;

                    for (const auto& bucket : buckets) {
                        if (bucket.keys.empty())
                            continue;

                        bool found_seed = false;
                        std::vector<uint64_t> proposed_slots;
                        proposed_slots.reserve(bucket.keys.size());

                        for (int attempt = 0; attempt < 1000000; attempt += 1) {
                            uint64_t seed = rng();
                            if (seed == 0)
                                seed = 1;

                            bool collision = false;
                            proposed_slots.clear();

                            for (uint64_t k : bucket.keys) {
                                uint64_t h = HashCore::hash_one_pass(k);
                                uint64_t s_idx = (h ^ seed) & res.slot_mask;

                                if (slot_used[s_idx]) {
                                    collision = true;
                                    break;
                                }
                                for (auto ps : proposed_slots)
                                    if (ps == s_idx) {
                                        collision = true;
                                        break;
                                    }
                                if (collision)
                                    break;
                                proposed_slots.push_back(s_idx);
                            }

                            if (!collision) {
                                res.control[bucket.id] = seed;
                                for (size_t i = 0; i < bucket.keys.size(); i += 1) {
                                    uint64_t k = bucket.keys[i];
                                    uint64_t s_idx = proposed_slots[i];
                                    slot_used[s_idx] = true;
                                    res.keys[s_idx] = k;

                                    for (const auto& input_kv : data) {
                                        if (input_kv.first == k) {
                                            res.values[s_idx] = input_kv.second;
                                            break;
                                        }
                                    }
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

                    slot_factor *= 1.3;
                    bucket_factor *= 1.1;
                    if (slot_factor > 10.0)
                        return {};
                }
            }
    };
} // namespace shm_pm

#pragma pack(pop)
#endif //HASH_HPP