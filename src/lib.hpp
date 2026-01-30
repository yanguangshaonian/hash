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
    constexpr uint64_t MAGIC_CODE = 0xDEADBEEFCAFEBABE;
    constexpr uint64_t EMPTY_KEY = ~0ULL;
    constexpr uint64_t GOLDEN_RATIO_64 = 0x9e3779b97f4a7c15ULL;

    inline uint64_t align_to_huge_page(uint64_t size) {
        return (size + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);
    }

    inline uint64_t align_to_cache_line(uint64_t size) {
        return (size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    }

    namespace HashCore {
        __attribute__((always_inline)) inline uint64_t hash_one_pass(uint64_t k) {
            return (k * GOLDEN_RATIO_64);
        }
    } // namespace HashCore

    // ----------------------------------------------------------------
    // 并发安全的数据单元
    // ----------------------------------------------------------------
    template<class T, size_t ALIGN> class alignas(ALIGN) PaddedSlot {
        public:
            uint64_t key;
            T value;
            // std::atomic<bool> busy_flag = false;

            // 指数退避自旋锁
            // inline __attribute__((always_inline)) bool lock(bool allow_fail = true) noexcept {
            //     while(true) {
            //         uint8_t delay = 0b0000'0001;
            //         while(busy_flag.load(std::memory_order_relaxed)) {
            //             for(auto i = 0; i < delay; i += 1) {
            //                 asm volatile("pause" ::: "memory");
            //             }
            //             if(delay < 0b1000'0000) {
            //                 delay <<= 1;
            //             } else {
            //                 if(allow_fail)
            //                     return false;
            //             }
            //         }
            //         if(!busy_flag.exchange(true, std::memory_order_acquire)) {
            //             return true;
            //         }
            //     }
            // }

            // inline __attribute__((always_inline)) void unlock() noexcept {
            //     busy_flag.store(false, std::memory_order_release);
            // }
    };

    // ----------------------------------------------------------------
    // 共享内存头部(Header)
    // ----------------------------------------------------------------
    class alignas(CACHE_LINE_SIZE) ShmMapHeader {
        public:
            volatile uint64_t magic;

            // 完美哈希参数
            uint64_t bucket_mask;
            uint64_t slot_mask;
            uint32_t bucket_shift;

            // 统计信息
            uint64_t item_count;
            uint64_t bucket_count;
            uint64_t slot_count;

            // 内存偏移量(相对于基地址)
            uint64_t offset_control;
            uint64_t offset_slots;

            // 文件元数据
            uint64_t total_file_size;
            uint64_t align_param; // 记录 ALIGN
            uint64_t value_size;  // 记录 sizeof(T)
            uint64_t slot_size;   // 记录 sizeof(PaddedSlot)
    };

    // ----------------------------------------------------------------
    // 映射视图(View) - 极速访问路径
    // ----------------------------------------------------------------
    template<class T, size_t ALIGN = CACHE_LINE_SIZE> class SharedMapView {
        private:
            const uint8_t* base_ptr = nullptr;
            const ShmMapHeader* header = nullptr;

            // 指针缓存
            const uint32_t* __restrict control_table = nullptr;
            const PaddedSlot<T, ALIGN>* __restrict slot_table = nullptr;

            // 本地缓存数据
            uint32_t local_bucket_shift = 0;
            uint64_t local_slot_mask = 0;

        public:
            void init(uint8_t* mapped_address) {
                base_ptr = mapped_address;
                header = reinterpret_cast<const ShmMapHeader*>(base_ptr);

                // 缓存热点元数据
                local_bucket_shift = header->bucket_shift;
                local_slot_mask = header->slot_mask;

                control_table = reinterpret_cast<const uint32_t*>(base_ptr + header->offset_control);
                // 直接定位到 Slot 数组
                slot_table = reinterpret_cast<const PaddedSlot<T, ALIGN>*>(base_ptr + header->offset_slots);
            }

            __attribute__((always_inline)) const PaddedSlot<T, ALIGN>* get(uint64_t key) const {
                auto h = HashCore::hash_one_pass(key);
                auto seed = control_table[h >> local_bucket_shift];
                auto slot_idx = (h ^ seed) & local_slot_mask;
                return &slot_table[slot_idx];
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
    template<class T, size_t ALIGN = CACHE_LINE_SIZE> class ShmMapStorage {
        private:
            enum class JoinResult {
                SUCCESS,
                FILE_NOT_FOUND, // 需要去创建
                DATA_CORRUPT,   // 文件存在但无效(Magic 错误或正在初始化)
                TYPE_MISMATCH,  // 数据结构版本不一致
                SYSTEM_ERROR    // mmap 失败等系统级错误
            };

            int32_t shm_fd = -1;
            uint8_t* mapped_ptr = nullptr;
            uint64_t mapped_size = 0;
            std::string storage_name;
            SharedMapView<T, ALIGN> view;

            using SlotType = PaddedSlot<T, ALIGN>;

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
                auto flags = MAP_SHARED;
                if (use_hugepage) {
                    flags |= MAP_HUGETLB;
                }
                auto ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, this->shm_fd, 0);
                if (ptr == MAP_FAILED) {
                    return nullptr;
                }
                return reinterpret_cast<uint8_t*>(ptr);
            }

            JoinResult try_join_existing() {
                this->shm_fd = shm_open(this->storage_name.c_str(), O_RDWR, 0660);
                if (this->shm_fd == -1) {
                    if (errno == ENOENT) {
                        return JoinResult::FILE_NOT_FOUND;
                    }
                    return JoinResult::SYSTEM_ERROR;
                }

                auto temp_ptr = map_memory_segment(MIN_MAP_SIZE, false);
                if (!temp_ptr) {
                    close(this->shm_fd);
                    return JoinResult::SYSTEM_ERROR;
                }

                auto header = reinterpret_cast<ShmMapHeader*>(temp_ptr);
                auto wait_count = 0;
                while (header->magic != MAGIC_CODE) {
                    wait_count += 1;
                    if (wait_count > 2000) {
                        log_msg("ERROR", "Magic 校验超时, 文件可能已损坏");
                        munmap(temp_ptr, MIN_MAP_SIZE);
                        close(this->shm_fd);
                        shm_unlink(this->storage_name.c_str()); // 清理
                        return JoinResult::DATA_CORRUPT;
                    }
                    std::this_thread::sleep_for(milliseconds(1));
                    std::atomic_thread_fence(std::memory_order_acquire);
                }

                // 校验结构(Key/Value是否合并)
                if (header->align_param != ALIGN || header->value_size != sizeof(T) ||
                    header->slot_size != sizeof(SlotType)) {
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
                    log_msg("WARN", "HugePage 映射失败, 尝试降级");
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

            class BuildResult {
                public:
                    bool success = false;
                    uint64_t bucket_mask = 0;
                    uint64_t slot_mask = 0;
                    uint32_t bucket_shift = 0;
                    std::vector<uint32_t> control;
                    std::vector<uint64_t> keys;
                    std::vector<size_t> value_indices;
            };

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

            // =========================================================
            //  数据校验
            // =========================================================
            bool check_data_validity(const std::vector<std::pair<uint64_t, T>>& data) {
                if (data.empty())
                    return true;

                std::vector<uint64_t> keys;
                keys.reserve(data.size());
                for (const auto& kv : data) {
                    keys.push_back(kv.first);
                }
                std::sort(keys.begin(), keys.end());

                // [修复] 独立检查 EMPTY_KEY(~0ULL 排序后一定在末尾)
                if (keys.back() == EMPTY_KEY) {
                    log_msg("ERROR", "输入数据包含保留的 EMPTY_KEY(~0ULL)，无法存储! ");
                    return false;
                }

                // 检查重复 Key
                for (size_t i = 1; i < keys.size(); i += 1) {
                    if (keys[i] == keys[i - 1]) {
                        std::stringstream ss;
                        ss << "发现重复 Key: " << keys[i];
                        log_msg("ERROR", ss.str());
                        return false;
                    }
                }

                // 检查哈希碰撞
                std::vector<uint64_t> hashes;
                hashes.reserve(data.size());
                for (uint64_t k : keys) {
                    hashes.push_back(HashCore::hash_one_pass(k));
                }
                std::sort(hashes.begin(), hashes.end());

                for (size_t i = 1; i < hashes.size(); i += 1) {
                    if (hashes[i] == hashes[i - 1]) {
                        log_msg("ERROR",
                                "致命错误: 检测到哈希碰撞! 不同的 Key 产生了相同的 Hash 值。请更换 Hash 函数。");
                        return false;
                    }
                }

                return true;
            }

            // =========================================================
            //  完美哈希构建(Fail Fast 策略)
            // =========================================================
            BuildResult build_perfect_hash_tables(const std::vector<std::pair<uint64_t, T>>& data) {
                auto n = data.size();
                if (n == 0)
                    return {};

                if (!check_data_validity(data)) {
                    return {};
                }

                BuildResult res;
                auto slot_factor = 1.15;
                auto bucket_factor = 0.95;

                // 桶信息
                class BucketInfo {
                    public:
                        uint64_t id;
                        std::vector<size_t> data_indices;
                };

                auto start_time = std::chrono::steady_clock::now();
                auto last_log_time = start_time;
                size_t attempt_count = 0;

                while (true) {
                    attempt_count += 1;
                    auto now = std::chrono::steady_clock::now();

                    // 每秒输出心跳日志
                    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 1) {
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        std::stringstream ss;
                        ss << "构建中... [耗时:" << elapsed << "s] [重试:" << attempt_count << "] "
                           << "[扩容:" << std::fixed << std::setprecision(2) << slot_factor << "x]";
                        log_msg("INFO", ss.str());
                        last_log_time = now;
                    }

                    // 计算表大小(2的幂次对齐)
                    uint64_t slot_cnt = next_pow2(static_cast<uint64_t>(n * slot_factor));
                    if (slot_cnt < 1024) {
                        slot_cnt = 1024;
                    }

                    uint64_t bucket_cnt = next_pow2(static_cast<uint64_t>(n * bucket_factor));
                    if (bucket_cnt < 4) {
                        bucket_cnt = 4;
                    }

                    res.bucket_mask = bucket_cnt - 1;
                    res.slot_mask = slot_cnt - 1;

                    auto bucket_bits = 0;
                    while ((1ULL << bucket_bits) < bucket_cnt) {
                        bucket_bits += 1;
                    }
                    res.bucket_shift = 64 - bucket_bits;

                    // 分桶(Mapping)
                    std::vector<BucketInfo> buckets(bucket_cnt);
                    for (uint64_t i = 0; i < bucket_cnt; i += 1) {
                        buckets[i].id = i;
                    }

                    for (size_t i = 0; i < n; i += 1) {
                        uint64_t h = HashCore::hash_one_pass(data[i].first);
                        buckets[h >> res.bucket_shift].data_indices.push_back(i);
                    }

                    // 优先处理元素多的大桶
                    std::sort(buckets.begin(), buckets.end(), [](const auto& a, const auto& b) {
                        return a.data_indices.size() > b.data_indices.size();
                    });

                    // 初始化结果集
                    res.control.assign(bucket_cnt, 0);
                    res.keys.assign(slot_cnt, EMPTY_KEY);
                    res.value_indices.assign(slot_cnt, SIZE_MAX);

                    std::vector<bool> slot_used(slot_cnt, false);

                    // 使用 attempt_count 扰动随机种子，确保每次重试都不一样
                    std::mt19937 rng(123456 + attempt_count);

                    bool success = true;

                    // 为每个桶寻找 Perfect Seed
                    for (const auto& bucket : buckets) {
                        if (bucket.data_indices.empty())
                            continue;

                        bool found_seed = false;
                        std::vector<uint64_t> proposed_slots;
                        proposed_slots.reserve(bucket.data_indices.size());

                        auto max_bucket_attempts = 1048576;

                        for (auto attempt = 0; attempt < max_bucket_attempts; attempt += 1) {
                            uint32_t seed = rng();
                            if (seed == 0) {
                                seed = 1;
                            }

                            bool collision = false;
                            proposed_slots.clear();

                            // 模拟放置桶内所有元素
                            for (size_t d_idx : bucket.data_indices) {
                                uint64_t k = data[d_idx].first;
                                // 核心映射公式
                                uint64_t s_idx = (HashCore::hash_one_pass(k) ^ seed) & res.slot_mask;

                                // 检查冲突: 位置已被占用 或 桶内自身冲突
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

                            // 找到无冲突 Seed
                            if (!collision) {
                                res.control[bucket.id] = seed;
                                // 真正落实到 Slot 表中
                                for (size_t i = 0; i < bucket.data_indices.size(); i += 1) {
                                    size_t original_idx = bucket.data_indices[i];
                                    uint64_t s_idx = proposed_slots[i];

                                    slot_used[s_idx] = true;
                                    res.keys[s_idx] = data[original_idx].first;
                                    res.value_indices[s_idx] = original_idx;
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
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                        if (elapsed > 100) { // 只有耗时显著才打印成功日志
                            std::stringstream ss;
                            ss << "完美哈希构建成功! 总耗时:" << elapsed << "ms, 最终空间因子:" << slot_factor << "x";
                            log_msg("INFO", ss.str());
                        }
                        return res;
                    }

                    // 每次微调 5%，逐步逼近最优解
                    slot_factor *= 1.05;
                    bucket_factor *= 1.05;

                    if (slot_factor > 5.0) {
                        log_msg("ERROR", "构建失败: 空间因子超过 5.0x 仍无法收敛，请检查数据分布是否极度畸形。");
                        return {};
                    }
                }
            }

            bool try_create_new(const std::vector<std::pair<uint64_t, T>>& data) {
                if (data.empty())
                    return false;
                auto build_res = build_perfect_hash_tables(data);
                if (!build_res.success) {
                    log_msg("ERROR", "完美哈希构建失败");
                    return false;
                }

                // 计算内存布局: Header + Control + Slots(Key+Value)
                size_t header_sz = align_to_cache_line(sizeof(ShmMapHeader));
                size_t ctrl_sz = align_to_cache_line(build_res.control.size() * sizeof(uint32_t));
                // Key 表消失了, 合并进了 Slot
                size_t slots_sz = align_to_cache_line(build_res.keys.size() * sizeof(SlotType));

                size_t total_sz = header_sz + ctrl_sz + slots_sz;
                size_t aligned_file_sz = align_to_huge_page(total_sz);

                size_t waste_sz = aligned_file_sz - total_sz;

                {
                    stringstream ss;
                    ss << "开始创建共享内存(Perfect Hash Map - AoS Optimized):\n"
                       << "  名称            = " << this->storage_name << "\n"
                       << "  逻辑数据量      = " << data.size() << "(实际写入)\n"
                       << "  物理槽位数量    = " << build_res.keys.size() << "(Slot Count, Load Factor ≈ " << std::fixed
                       << std::setprecision(2) << ((double) data.size() / build_res.keys.size()) << ")\n"
                       << "  sizeof(T)       = " << sizeof(T)
                       << " 字节\n"
                       // 注意: 原 PaddedValue 现为 SlotType(HashSlot)
                       << "  sizeof(Slot)    = " << sizeof(SlotType) << " 字节\n"
                       << "  alignof(T)      = " << alignof(T) << "\n"
                       << "  alignof(Slot)   = " << alignof(SlotType) << "(CacheLine=" << ALIGN << ")";
                    log_msg("INFO", ss.str());
                }

                {
                    stringstream ss;
                    ss << "内存布局计算详情:\n"
                       << "  [1] Header 区     = " << header_sz << " 字节\n"
                       << "  [2] Control 表    = " << ctrl_sz
                       << " 字节\n"
                       // 原 Key 表 和 Value 表合并为 Slot 表
                       << "  [3] Slot 表       = " << slots_sz << " 字节\n"
                       << "  --------------------------------\n"
                       << "  实际数据总大小    = " << total_sz << " 字节\n"
                       << "  HugePage对齐后    = " << aligned_file_sz << " 字节\n"
                       << "  对齐浪费空间      = " << waste_sz << " 字节(" << std::fixed << std::setprecision(2)
                       << (100.0 * waste_sz / aligned_file_sz) << "%)";
                    log_msg("INFO", ss.str());
                }

                this->shm_fd = shm_open(this->storage_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0660);
                if (this->shm_fd == -1)
                    return false;
                if (ftruncate(this->shm_fd, aligned_file_sz) == -1) {
                    close(this->shm_fd);
                    shm_unlink(this->storage_name.c_str());
                    return false;
                }

                this->mapped_ptr = map_memory_segment(aligned_file_sz, true);
                if (!this->mapped_ptr) {
                    this->mapped_ptr = map_memory_segment(aligned_file_sz, false);
                    if (!this->mapped_ptr) {
                        close(this->shm_fd);
                        shm_unlink(this->storage_name.c_str());
                        return false;
                    }
                }
                this->mapped_size = aligned_file_sz;

                // 填充 Header
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
                hdr->slot_size = sizeof(SlotType);

                hdr->offset_control = header_sz;
                hdr->offset_slots = header_sz + ctrl_sz; // 直接指向 Slot 数组

                // 写入 Control
                std::memcpy(this->mapped_ptr + hdr->offset_control, build_res.control.data(),
                            build_res.control.size() * sizeof(uint32_t));

                // 写入 Slots(Key + Value)
                auto* slot_ptr = reinterpret_cast<SlotType*>(this->mapped_ptr + hdr->offset_slots);

                // 初始化所有 Slot 为空 Key
                for (size_t i = 0; i < hdr->slot_count; i += 1) {
                    // 使用 placement new 初始化对齐内存
                    new (&slot_ptr[i]) SlotType();
                    slot_ptr[i].key = EMPTY_KEY;
                    // slot_ptr[i].unlock();
                }

                for (size_t i = 0; i < build_res.value_indices.size(); i += 1) {
                    size_t src_idx = build_res.value_indices[i];
                    if (src_idx != SIZE_MAX) {
                        // 写入 Key 和 Value
                        slot_ptr[i].key = data[src_idx].first;
                        slot_ptr[i].value = data[src_idx].second;
                    }
                }

                std::atomic_thread_fence(std::memory_order_release);
                hdr->magic = MAGIC_CODE;

                this->view.init(this->mapped_ptr);
                {
                    stringstream ss;
                    ss << "共享内存创建成功并已发布:\n"
                       << "  映射总大小        = " << aligned_file_sz << " 字节(" << std::fixed << std::setprecision(2)
                       << (aligned_file_sz / 1024.0 / 1024.0) << " MB)";
                    log_msg("INFO", ss.str());
                }
                return true;
            }

        public:
            bool build(std::string name, const std::vector<std::pair<uint64_t, T>>& init_data = {}) {
                if (geteuid() != 0)
                    throw std::runtime_error("需要 Root 权限(HugePage)");
                this->storage_name = name;

                for (auto i = 0; i < 3; i += 1) {
                    auto join_res = try_join_existing();
                    if (join_res == JoinResult::SUCCESS)
                        return true;
                    if (join_res == JoinResult::TYPE_MISMATCH)
                        throw std::runtime_error("数据结构版本不匹配");
                    if (join_res == JoinResult::DATA_CORRUPT) {
                        log_msg("WARN", "文件损坏, 重置中...");
                        continue;
                    }
                    if (try_create_new(init_data))
                        return false;
                    if (errno == EEXIST) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        continue;
                    }
                    throw std::runtime_error("初始化失败: " + string(strerror(errno)));
                }
                throw std::runtime_error("初始化超时");
            }

            inline SharedMapView<T, ALIGN>& get_view() {
                return this->view;
            }

            ~ShmMapStorage() {
                if (this->mapped_ptr)
                    munmap(this->mapped_ptr, this->mapped_size);
                if (shm_fd != -1)
                    close(shm_fd);
            }
    };
} // namespace shm_pm

#pragma pack(pop)
#endif // HASH_STORAGE_HPP