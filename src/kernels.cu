#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix in GPU.
 * 
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 * 
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param d_input A flattened matrix of size rows * cols.
 * @param partial_sum 用于存储中间计算结果
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return 没有
 * */ 
template <typename T>
__global__ void trace_kernel(const T* d_input, T* partial_sum, size_t rows, size_t cols) {
  extern __shared__ char shared_mem[]; // 在kernel被调用时再传入
  T* sdata = reinterpret_cast<T*>(shared_mem);

  const size_t tid = threadIdx.x;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t diag_len = min(cols, rows);

  // 初始化共享内存
  sdata[tid] = T{0};
  if (i < diag_len) sdata[tid] = d_input[i * cols + i];
  __syncthreads();

  // 归并求得当前block的和
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid+s];
    }
    __syncthreads();
  }

  // 写回block的和
  if (tid == 0) {
    partial_sum[blockIdx.x] = sdata[0];
  }
}


/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  if (h_input.size() == 0) return T{0};

  const size_t diag_len = std::min(rows, cols);
  if (diag_len == 0) return T{0};

  // 分配设备内存
  const size_t input_size = rows*cols;
  const size_t input_bytes = input_size * sizeof(T);

  T* d_input = nullptr;
  T* d_partial = nullptr;

  cudaMalloc(&d_input, input_bytes);
  cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

  // 设备设置
  const int block_size = 256;
  const int grid_size = (static_cast<int>(diag_len) + block_size - 1) / block_size;

  // 分配部分求和结果的内存
  cudaMalloc(&d_partial, grid_size * sizeof(T));

  // 启动kernel
  const size_t shared_mem_size = block_size * sizeof(T);
  trace_kernel<<<grid_size, block_size, shared_mem_size>>>(
    d_input, d_partial, rows, cols
  );

  // 复制部分求和结果
  std::vector<T> h_partial(grid_size);
  cudaDeviceSynchronize();
  cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

  // 最终求和
  T total = T{0};
  for (T val : h_partial) total += val;

  cudaFree(d_input);
  cudaFree(d_partial);

  return total;

}

// T->float
template <typename T>
__device__ __host__ inline float to_float(T val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}
// T/double -> float
template <typename T>
__device__ __host__ inline T from_float(float val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else {
        return static_cast<T>(val);
    }
}

/**
 * @brief 执行 FlashAttention 前向计算，支持分组查询注意力（GQA）和因果掩码
 * blockDim.x=Br*32
 *
 *
 * 所有张量必须为连续内存（contiguous），按行优先（C-order）展开。
 *
 * @tparam T                输入/输出张量的数据类型（如 float 或 __half）
 *
 * @param[in]  q            Query 张量，形状 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in]  k            Key 张量，形状 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in]  v            Value 张量，形状 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] o            输出张量，形状 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in,out] m         Softmax 最大值缓冲区，形状 [batch_size, query_heads, tgt_seq_len]
 * @param[in,out] l         Softmax 归一化因子缓冲区，形状 [batch_size, query_heads, tgt_seq_len]
 * @param[in]  batch_size   批次大小
 * @param[in]  tgt_seq_len  目标序列长度（Query 序列长度）
 * @param[in]  src_seq_len  源序列长度（Key/Value 序列长度）
 * @param[in]  query_heads  Query 头的数量
 * @param[in]  kv_heads     Key/Value 头的数量（必须整除 query_heads）
 * @param[in]  head_dim     每个注意力头的维度
 * @param[in]  softmax_scale 缩放因子，通常为 1.0f / sqrtf(head_dim)
 * @param[in]  is_causal    是否启用因果掩码（true 表示仅允许 j <= i 的注意力）
 * @param[in]  Br           每次处理的 Query 分块行数（目标序列方向的 tile 高度）
 * @param[in]  Bc           每次加载的 Key/Value 分块行数（源序列方向的 tile 高度）
 */
template <typename T, int Br, int Bc>
__global__ void flashAttention_(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    double* __restrict__ m,
    double* __restrict__ l,
    const int batch_size,
    const int target_seq_len,
    const int src_seq_len,
    const int query_heads,
    const int kv_heads,
    const int head_dim,          // <= 64
    const bool is_causal
) {
    // 1. 解析 block 索引
    const int batch_qhead = blockIdx.y;
    const int batch_id = batch_qhead / query_heads; 
    const int q_head_id = batch_qhead % query_heads; 

    const int q_tile_start = blockIdx.x * Br;
    const int q_tile_end = min(q_tile_start + Br, target_seq_len);

    const int warp_id = threadIdx.x / 32;   // [0, Br)
    const int lane_id = threadIdx.x % 32;   // [0, 31]

    const int q_pos = q_tile_start + warp_id;
    const bool valid_query = (q_pos < target_seq_len);

    // GQA: map query head to kv head
    const int group_size = query_heads / kv_heads;
    const int kv_head_id = q_head_id / group_size;

    // softmax scale (建议作为参数传入，但这里按你要求计算)
    const double softmax_scale = 1.0 / sqrt(static_cast<double>(head_dim));

    // 2. Shared memory 布局
    extern __shared__ char s_mem[];
    T* s_Q = reinterpret_cast<T*>(s_mem);
    T* s_K = s_Q + Br * head_dim;
    T* s_V = s_K + Bc * head_dim;

    // 3. 加载 Q tile 到 shared memory
    #pragma unroll
    for (int idx = threadIdx.x; idx < Br * head_dim; idx += blockDim.x) {
        const int local_q = idx / head_dim;
        const int d_idx = idx % head_dim;
        const int global_q = q_tile_start + local_q;

        T val = T(0);
        if (global_q < target_seq_len) {
            const int64_t q_offset = ((int64_t)batch_id * target_seq_len + global_q) * query_heads + q_head_id;
            val = q[q_offset * head_dim + d_idx];
        }
        s_Q[idx] = val;
    }
    __syncthreads();

    // 4. 初始化 softmax 状态 (m, l)
    double row_m_prev = -__DBL_MAX__;
    double row_l_prev = 0.0;

    if (valid_query) {
        const int64_t ml_offset = ((int64_t)batch_id * query_heads + q_head_id) * target_seq_len + q_pos;
        row_m_prev = m[ml_offset];
        row_l_prev = l[ml_offset];
    }

    // 5. 初始化输出累加器 (寄存器，head_dim <= 64)
    double acc_o[64]; // 最多64维
    for (int d = 0; d < head_dim; ++d) {
        acc_o[d] = 0.0;
    }

    // 6. 沿 src_seq_len 分块处理 K/V
    #pragma unroll
    for (int k_start = 0; k_start < src_seq_len; k_start += Bc) {
        const int k_tile_end = min(k_start + Bc, src_seq_len);
        const int actual_Bc = k_tile_end - k_start;

        // 6.1 加载 K/V tile
        #pragma unroll
        for (int idx = threadIdx.x; idx < Bc * head_dim; idx += blockDim.x) {
            const int local_k = idx / head_dim;
            const int d_idx = idx % head_dim;
            const int global_k = k_start + local_k;

            T k_val = T(0), v_val = T(0);
            if (global_k < src_seq_len) {
                const int64_t kv_offset = ((int64_t)batch_id * src_seq_len + global_k) * kv_heads + kv_head_id;
                k_val = k[kv_offset * head_dim + d_idx];
                v_val = v[kv_offset * head_dim + d_idx];
            }
            s_K[idx] = k_val;
            s_V[idx] = v_val;
        }
        __syncthreads();

        // 6.2 计算局部 logits 和 max (第一遍)
        double local_m = -__DBL_MAX__;
        double local_p[Bc]; // logits or weights

        if (valid_query) {
            #pragma unroll
            for (int local_k = 0; local_k < actual_Bc; ++local_k) {
                const int k_pos = k_start + local_k;

                // Causal mask
                if (is_causal && k_pos > q_pos) {
                    local_p[local_k] = -__DBL_MAX__; // 设为极小值
                    continue;
                }

                float qk = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    qk += static_cast<double>(to_float(s_Q[warp_id * head_dim + d])) *
                          static_cast<double>(to_float(s_K[local_k * head_dim + d]));
                }
                qk *= softmax_scale;
                local_p[local_k] = qk;

                if (qk > local_m) {
                    local_m = qk;
                }
            }

            // 6.3 计算局部 softmax 权重 (第二遍)
            double local_l = 0.0;
            #pragma unroll
            for (int local_k = 0; local_k < actual_Bc; ++local_k) {
                if (is_causal && (k_start + local_k) > q_pos) {
                    local_p[local_k] = 0.0;
                    continue;
                }
                double weight = exp(local_p[local_k] - local_m);
                local_p[local_k] = weight;
                local_l += weight;
            }

            // 6.4 合并状态：MDFOp
            const double new_m = fmax(row_m_prev, local_m);
            const double new_l = row_l_prev * exp(row_m_prev - new_m) + local_l * exp(local_m - new_m);

            // 6.5 更新输出累加器
            const double coeff_prev = (row_l_prev > 0) ? (row_l_prev * exp(row_m_prev - new_m) / new_l) : 0.0;
            const double coeff_curr = (local_l > 0) ? (exp(local_m - new_m) / new_l) : 0.0;

            // acc_o = acc_o * coeff_prev + sum(weight * V) * coeff_curr
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                double pv_sum = 0.0;
                #pragma unroll
                for (int local_k = 0; local_k < actual_Bc; ++local_k) {
                    if (is_causal && (k_start + local_k) > q_pos) continue;
                    pv_sum += local_p[local_k] * static_cast<double>(to_float(s_V[local_k * head_dim + d]));
                }
                acc_o[d] = acc_o[d] * coeff_prev + pv_sum * coeff_curr;
            }

            // 更新状态
            row_m_prev = new_m;
            row_l_prev = new_l;
        }
        __syncthreads();
    }

    // 7. 写回最终结果
    if (valid_query) {
        // 写回 m/l
        const int64_t ml_offset = ((int64_t)batch_id * query_heads + q_head_id) * target_seq_len + q_pos;
        m[ml_offset] = row_m_prev;
        l[ml_offset] = row_l_prev;

        // 写回 O
        const int64_t o_offset = ((int64_t)batch_id * target_seq_len + q_pos) * query_heads + q_head_id;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            o[o_offset * head_dim + d] = from_float<T>(static_cast<float>(acc_o[d]));
        }
    }
}



/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(
    const std::vector<T>& h_q,
    const std::vector<T>& h_k,
    const std::vector<T>& h_v,
    std::vector<T>& h_o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal
) {

    // === 2. 计算张量尺寸 ===
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = k_size;
    size_t o_size = q_size;

    // === 3. 分配设备内存 ===
    T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    double *d_m = nullptr, *d_l = nullptr;

    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));

    // m/l: [batch_size][query_heads][target_seq_len]
    size_t ml_size = batch_size * query_heads * target_seq_len;
    cudaMalloc(&d_m, ml_size * sizeof(double));
    cudaMalloc(&d_l, ml_size * sizeof(double));

    // === 4. 拷贝 Q/K/V 到设备 ===
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);

    // === 5. 初始化 m/l ===
    // m = -inf, l = 0
    cudaMemset(d_m, 0xFF, ml_size * sizeof(double)); // -inf in double (all bits set)
    cudaMemset(d_l, 0,    ml_size * sizeof(double));

    // === 6. 启动配置（FlashAttention v2 风格）===
    const int Br = 32;  // Query 分块大小（可调，通常 16~32）
    const int Bc = 64;  // Key/Value 分块大小（可调，通常 32~128）

    // Grid: 
    //   x: number of query tiles = ceil(target_seq_len / Br)
    //   y: number of (batch, query_head) pairs = batch_size * query_heads
    dim3 grid(
        (target_seq_len + Br - 1) / Br,
        batch_size * query_heads
    );

    // Block: Br warps → Br * 32 threads
    dim3 block(Br * 32);

    // Shared memory: s_Q[Br*head_dim] + s_K[Bc*head_dim] + s_V[Bc*head_dim]
    size_t shared_mem_size = (Br + 2 * Bc) * head_dim * sizeof(T);

    // === 7. 启动 kernel ===
    flashAttention_<T, Br, Bc><<<grid, block, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        d_m, d_l,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal
    );

    // === 8. 同步并检查错误 ===
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // 清理内存
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
        cudaFree(d_m); cudaFree(d_l);
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    // === 9. 拷贝结果回 host ===
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);

    // === 10. 释放设备内存 ===
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_m);
    cudaFree(d_l);
}




















// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
