
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>

/**
 * 模拟吵闹邻居：持续进行 L3 Cache Thrashing (缓存抖动)
 * 编译: gcc -O3 noisy_neighbor.c -o noisy_neighbor
 */

// 假设 L3 为 32MB，我们分配 64MB 以确保完全覆盖并淘汰旧数据
#define L3_SIZE_BYTES (64 * 1024 * 1024)
#define CACHE_LINE_SIZE 64

int main(int argc, char **argv) {
    // 1. 将进程绑定到核心 0，以干扰共享 L3 的其他兄弟核心
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
        perror("sched_setaffinity");
    }

    // 2. 分配大内存块 (使用对齐分配以模拟真实高性能场景)
    void* buffer = NULL;
    if (posix_memalign(&buffer, 4096, L3_SIZE_BYTES) != 0) {
        perror("posix_memalign");
        return 1;
    }

    // 初始化内存，防止首次访问时的缺页中断影响性能测试
    memset(buffer, 0xAA, L3_SIZE_BYTES);

    volatile uint8_t* data = (volatile uint8_t*)buffer;
    printf("开始模拟吵闹邻居，持续占用 L3 缓存...\n");

    // 3. 核心循环：不断进行线性读写，跨步长为 Cache Line 大小
    while (1) {
        for (size_t i = 0; i < L3_SIZE_BYTES; i += CACHE_LINE_SIZE) {
            // 读出数据并写回，强制触发 Cache Line 的修改和写回流
            uint8_t temp = data[i];
            data[i] = temp + (uint8_t)1; // 禁止使用自增运算符 (遵照偏好)
        }
    }

    free(buffer);
    return 0;
}