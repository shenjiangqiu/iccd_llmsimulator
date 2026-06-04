# PIM GEMV 对 2-bit 量化 KV 的计算问题

## 1. 背景：PIM GEMV 的 2-bit 量化分解

在 PIM 中做 2-bit 量化 GEMV 时，利用非对称量化公式：

$$Q \cdot K = s_Q s_K \left[\sum\hat{q}\hat{k} - z_K\sum\hat{q} - z_Q\sum\hat{k} + d z_Q z_K\right]$$

这要求**一组 2-bit 元素共享同一个 scale $s$ 和 zero-point $z$**，才能把它们的 $\sum\hat{q}\hat{k}$、$\sum\hat{q}$、$\sum\hat{k}$ 在 2-bit PE 上累加，最后统一缩放。

共享 scale/z 的元素数量取决于**量化 group size**。例如 group size = 32，则每 32 个元素共享一组 $(s, z)$。

## 2. PIM 对量化方向的要求

### 2.1 Score computation ($S = Q \cdot K^\top$)

Q@K 是沿 **channel 维度**做内积：
$$S_{ij} = \sum_{c=1}^{d} Q_{ic} \cdot K_{jc}$$

每个内积把 $d$ 个 channel 上的 $Q \cdot K$ 乘积累加起来。如果 K 的 **channel 方向**上每 32 个元素共享 scale/z，那么这 32 个 channel 的 $\sum \hat{q}\hat{k}_{32}$ 可以在 2-bit PE 上先累加，最后统一缩放。

$$\downarrow \text{要求：K 必须做 per-token 量化（token 维度 group）}$$

```
K 矩阵:  tokens ──→
       ┌─────────────────────┐
       │ ch1 ch2 ... ch32 │ ch33 ... │  ← 每 32 channel 一组
       │ ←── group ──→    │          │     共享 scale/z
       ├─────────────────────┤
       │ ch1 ch2 ... ch32 │ ch33 ... │  ← 另一组 token 独立
       │ ←── group ──→    │          │
ch ↑   └─────────────────────┘
```

### 2.2 Context aggregation ($O = S \cdot V$)

Score@V 是沿 **token 维度**做加权和：
$$O_{ic} = \sum_{j=1}^{N} S_{ij} \cdot V_{jc}$$

每个输出元素把 $N$ 个 token 的 $S \cdot V$ 乘积累加起来。如果 V 的 **token 方向**上每 32 个元素共享 scale/z，那么这 32 个 token 的 V 值可以先在 2-bit PE 上累加，再统一缩放。

$$\downarrow \text{要求：V 必须做 per-channel 量化（channel 维度 group）}$$

```
V 矩阵:  tokens ──→
       ┌─────────────────────┐
       │  ch1                 │
       ├─────────────────────┤  ← 每 32 token 一组
ch ↑   │ tok1 tok2 ... tok32 │     共享 scale/z
       │ ←── group ──→       │
       ├─────────────────────┤
       │  ch2                 │
       └─────────────────────┘
```

## 3. 与 KIVI 的矛盾

KIVI 的量化方向恰好**相反**：

| | KIVI 方案 | PIM 友好方案 | 原因 |
|---|---|---|---|
| **K** | per-channel | **per-token** | K 有 outlier channel → per-channel 隔离异常值；但 PIM Q@K 需要沿 channel 累加 |
| **V** | per-token | **per-channel** | V 是 mixer → per-token 隔离误差传播；但 PIM Score@V 需要沿 token 累加 |

KIVI 的选择是从**精度**出发：per-channel K 把异常 channel 的误差锁在该 channel 内，per-token V 防止一个 token 的量化误差通过加权和扩散到其他 token。

PIM 友好方案是从**硬件效率**出发：shared scale/z 让 2-bit PE 可以在组内做无精度损失的累加。

## 4. 解决方案：先截取 outlier，再做 PIM 友好量化

核心思路不是修改量化方向，而是**把 outlier 单独抽出来**，让剩余的主体矩阵采用 PIM 友好的量化方向。

### 4.1 K 矩阵的处理

K 是一个 $\text{SeqLen} \times d$ 的二维矩阵。KIVI 发现某些 channel 的值特别大（outlier channel），需要 per-channel 量化来隔离误差。但 per-channel 量化与 PIM 的 per-token 要求矛盾。

**步骤**：
1. **离线识别**：对 K 矩阵沿 channel 维度做统计分析，找出数值范围显著大于其他 channel 的 outlier channel（通常 < 5%）。
2. **截取分离**：把 K 矩阵按 channel 拆成两部分：
   - $K_{\text{outlier}}$：outlier channel，大小 $\text{SeqLen} \times d_{\text{outlier}}$。保留 FP16 不做量化，或单独做 per-channel 量化。
   - $K_{\text{normal}}$：剩余 normal channel，大小 $\text{SeqLen} \times d_{\text{normal}}$（$d_{\text{normal}} \gg d_{\text{outlier}}$）。**做 per-token 量化**，每 32 个 channel 共享 scale/z，直接送给 PIM 做 2-bit 累加。
3. **混合计算**：
   - PIM 用 2-bit PE 计算 $Q_{\text{normal}} \cdot K_{\text{normal}}^\top$（主体，90%+ 的计算量）
   - GPU 用 FP16 计算 $Q_{\text{outlier}} \cdot K_{\text{outlier}}^\top$（少量 outlier channel）
   - 最终 score = PIM 结果（去量化后）+ GPU 结果

```
K 矩阵:    channel ──→
       ┌──────┬──────────────────────┐
       │outlier│   normal channels    │
       │ chs  │  (per-token 量化)    │
       │(FP16)│  group size = 32     │
       │      │  每 32 ch 共享 s,z    │
SeqLen │  GPU │       PIM            │
  ↑    │ path │       path           │
       └──────┴──────────────────────┘
```

### 4.2 V 矩阵的处理

V 同样是 $\text{SeqLen} \times d$ 矩阵。按 KIVI 的分析，V 需要 per-token 量化来隔离 token 间误差传播。但 per-token 量化与 PIM 的 per-channel 要求矛盾。

**步骤**：
1. **离线识别**：对 V 矩阵沿 token 维度做统计分析，找出 outlier token。
2. **截取分离**：
   - $V_{\text{outlier}}$：outlier token，大小 $\text{SeqLen}_{\text{outlier}} \times d$。保留 FP16 或 per-token 量化。
   - $V_{\text{normal}}$：剩余 normal token，大小 $\text{SeqLen}_{\text{normal}} \times d$。**做 per-channel 量化**，每 32 个 token 共享 scale/z，直接送给 PIM 做 2-bit 累加。
3. **混合计算**：
   - PIM 用 2-bit PE 计算 $S_{\text{normal}} \cdot V_{\text{normal}}$（主体）
   - GPU 用 FP16 计算 $S_{\text{outlier}} \cdot V_{\text{outlier}}$（少量 outlier token）
   - 最终 output = PIM 结果 + GPU 结果

```
V 矩阵:    channel ──→
       ┌──────────────────────────────┐
       │   normal tokens              │
       │  (per-channel 量化)          │ ← PIM path
       │   每 32 token 共享 s,z       │
       ├──────────────────────────────┤
       │   outlier tokens             │
       │   (FP16)                     │ ← GPU path
SeqLen └──────────────────────────────┘
  ↑
```

### 4.3 优势

- **精度保有**：outlier 部分走高精度路径，不损失精度
- **硬件高效**：主体矩阵（90%+ 的数据）使用 PIM 友好的量化方向，2-bit PE 直接计算
- **低开销**：outlier 比例小（通常 < 5%），GPU fallback 的计算量几乎可忽略
- **渐进式**：可以在不改变 KIVI 主流量化框架的情况下，在硬件层面增加这种截取分离策略

## 5. 对模拟器的影响

当前模拟器假设 K 和 V 使用统一的量化参数（$z_Q, z_K$）且量化方向一致。要实现截取分离策略需要：

- K/V cache 增加 outlier mask（标记哪些 channel/token 是 outlier）
- `computeGEMVLatency` 对 normal 部分使用 2-bit PE 时间，对 outlier 部分使用 GPU/FP16 时间
- 增加 GPU fallback 计算路径，将两部分结果相加
