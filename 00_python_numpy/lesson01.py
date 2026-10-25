"""
第 01 课：NumPy 基础 —— 形状、索引、axis、广播
=================================================

怎么用这个文件：
  1. 从上往下读，把每个  # TODO  的地方自己写出来
  2. 写完运行：  python3 lesson01.py
  3. 看到 [PASS] 才算过；看到 [FAIL] 就回去改，卡住了把你的写法发给我

这个文件里没有答案，只有检查。不要跳过 TODO。

本课要建立的直觉（后面所有网络都靠它）：
  * 张量的「形状」比它的数值更重要 —— 90% 的 bug 是形状不对
  * axis=k 的意思是「把第 k 个维度压掉」
  * 广播：形状不同的数组怎么自动对齐相加
"""

import numpy as np


# ============================================================
# 练习 1：创建数组与查看形状
# ============================================================
def ex1_create():
    # 1.1 创建一个 shape 为 (3, 4)、全是 0 的 float 数组 -> a
    a = np.zeros((3,4), dtype = float)
    # 1.2 创建一个 shape 为 (3, 4) 的数组，内容依次是 0,1,2,...,11 -> b
    #     提示：np.arange(12) 然后 reshape
    b = None
    b = np.arange(12).reshape(3,4)
    # 1.3 创建一个 shape 为 (2, 3)、元素服从标准正态分布的数组 -> c
    #     提示：np.random.default_rng(0).standard_normal(...)
    c = np.random.default_rng(0).standard_normal((2,3))

    return a, b, c


def check1():
    a, b, c = ex1_create()
    assert a.shape == (3, 4), f"1.1 形状应为 (3,4)，实际 {a.shape}"
    assert a.dtype == np.float64, f"1.1 应为 float，实际 {a.dtype}"
    assert np.all(a == 0), "1.1 应全为 0"
    assert b.shape == (3, 4), f"1.2 形状应为 (3,4)，实际 {b.shape}"
    assert b[0, 0] == 0 and b[1, 2] == 6 and b[2, 3] == 11, f"1.2 内容不对：\n{b}"
    assert c.shape == (2, 3), f"1.3 形状应为 (2,3)，实际 {c.shape}"


# ============================================================
# 练习 2：索引与切片
# ============================================================
M = np.arange(20).reshape(4, 5)   # 先运行一下看看它长什么样


def ex2_index():
    # 2.1 取第 2 行（索引为 1 的那一行）-> r       期望 shape: (5,)
    r = M[1, :]

    # 2.2 取第 3 列（索引为 2 的那一列）-> col     期望 shape: (4,)
    col = M[:, 2]

    # 2.3 取左上角 2x3 的子矩阵 -> sub              期望 shape: (2,3)
    sub = M[:2, :3]

    # 2.4 取第 0、2 行 -> evens                    期望 shape: (2,5)
    #     提示：切片支持步长 M[::2]
    evens = M[::2]

    return r, col, sub, evens


def check2():
    r, col, sub, evens = ex2_index()
    assert r is not None and r.shape == (5,) and np.all(r == [5, 6, 7, 8, 9]), f"2.1 不对：{r}"
    assert col is not None and col.shape == (4,) and np.all(col == [2, 7, 12, 17]), f"2.2 不对：{col}"
    assert sub is not None and sub.shape == (2, 3) and np.all(sub == [[0, 1, 2], [5, 6, 7]]), f"2.3 不对：{sub}"
    assert evens is not None and evens.shape == (2, 5) and np.all(evens == [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]]), f"2.4 不对：{evens}"


# ============================================================
# 练习 3：axis 归约 —— 新手最大的坑，务必想清楚
# ============================================================
X = np.arange(12).reshape(3, 4)   # 3 行 4 列


def ex3_axis():
    # 3.1 所有元素求和 -> s_all   （结果是一个标量）
    s_all = X.sum()

    # 3.2 沿 axis=0 求和 -> s0
    #     写下你的预测形状：__________   （提示：axis=0 就是「把第 0 维压掉」）
    s0 = X.sum(axis = 0)

    # 3.3 沿 axis=1 求和 -> s1
    #     写下你的预测形状：__________
    s1 = X.sum(axis = 1)

    # 3.4 沿 axis=0 求每列最大值 -> mx0
    mx0 = X.max(axis = 0)

    return s_all, s0, s1, mx0


def check3():
    s_all, s0, s1, mx0 = ex3_axis()
    assert s_all == 66, f"3.1 应为 66，实际 {s_all}"
    assert s0 is not None and s0.shape == (4,), f"3.2 形状应为 (4,)，实际 {getattr(s0, 'shape', None)}"
    assert np.all(s0 == [12, 15, 18, 21]), f"3.2 数值不对：{s0}"
    assert s1 is not None and s1.shape == (3,), f"3.3 形状应为 (3,)，实际 {getattr(s1, 'shape', None)}"
    assert np.all(s1 == [6, 22, 38]), f"3.3 数值不对：{s1}"
    assert mx0 is not None and np.all(mx0 == [8, 9, 10, 11]), f"3.4 数值不对：{mx0}"


# ============================================================
# 练习 4：广播 —— 两个形状不同的数组怎么相加
# ============================================================
def ex4_broadcast():
    # 4.1 造一个 shape (32, 1) 的列向量 col，和一个 shape (1, 10) 的行向量 row，
    #     让它们相加 -> bc
    #     先写下你的预测形状：_____32,10_____
    col = np.ones((32, 1))
    row = np.ones((1, 10))
    bc = col + row

    # 4.2 让 shape (32, 10) 的矩阵 MM 的「每一行减去该行的均值」-> centered
    #     提示：如果写成 MM - MM.mean(axis=1)，形状是 (32,10) - (32,)，会报错。
    #           查一下 mean 的 keepdims 参数怎么用。
    MM = np.random.default_rng(0).standard_normal((32, 10))
    centered = MM - MM.mean(axis = 1, keepdims = True)

    return bc, centered


def check4():
    bc, centered = ex4_broadcast()
    assert bc is not None and bc.shape == (32, 10), f"4.1 广播结果形状应为 (32,10)，实际 {getattr(bc, 'shape', None)}"
    assert centered is not None and centered.shape == (32, 10), f"4.2 形状应为 (32,10)，实际 {getattr(centered, 'shape', None)}"
    assert np.allclose(centered.mean(axis=1), 0, atol=1e-10), "4.2 每行减去均值后，每行均值应为 0"


# ============================================================
# 练习 5：矩阵乘法 —— 神经网络里最核心的运算
# ============================================================
def my_matmul(A, B):
    """不用 @、不用 np.dot，用循环实现矩阵乘法 C = A @ B。

    A: shape (m, n)
    B: shape (n, p)
    C: shape (m, p)，其中 C[i, j] = sum_k A[i, k] * B[k, j]

    步骤建议：
      1. m, n = A.shape; n2, p = B.shape
      2. assert n == n2，否则矩阵不能相乘
      3. 造一个 (m, p) 的全零 C
      4. 三重循环填进去
    """
    # TODO: 
    (m, n) = A.shape
    (n2, p) = B.shape
    assert n == n2
    C = np.zeros((m,p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C
    raise NotImplementedError("还没实现 my_matmul")


def ex5_matmul():
    A = np.arange(6).reshape(2, 3)          # (2,3)
    B = np.arange(12).reshape(3, 4)         # (3,4)

    # 5.1 用 @ 运算符求 C -> c_operator
    c_operator = A @ B

    # 5.2 调用你自己写的 my_matmul -> c_mine
    c_mine = my_matmul(A, B)

    # 5.3 故意算一个不合法的乘法 (2,3) @ (2,4)，把报错的文字打印出来
    #     提示：用 try / except Exception as e: print(e)
    #     这类报错信息以后你会天天看到，先熟悉一下它的措辞
    try:
        _ = A @ np.zeros((2, 4))
    except Exception as e:
        print("    [5.3 观察到的报错] ", e)

    return c_operator, c_mine


def check5():
    c_operator, c_mine = ex5_matmul()
    expected = np.array([[20, 23, 26, 29],
                         [56, 68, 80, 92]])
    assert c_operator is not None and np.all(c_operator == expected), f"5.1 不对：\n{c_operator}"
    assert np.all(c_mine == expected), f"5.2 你手写的 matmul 结果不对：\n{c_mine}"


# ============================================================
# 练习 6：综合 —— 手写数值稳定的 softmax
# ============================================================
def softmax(x, axis=-1):
    """softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    要求「数值稳定」：先把 x 减去它沿 axis 的最大值，再取 exp。
    为什么？（想一想：如果 x 里有 1000，np.exp(1000) 会变成什么？）

    步骤：
      1. x_max = x 沿 axis 的最大值（注意保留维度，否则广播会出错）
      2. e = np.exp(x - x_max)
      3. return e / e 沿 axis 求和
    """
    # TODO: 
    x_max = x.max(axis, keepdims = True)
    e = np.exp(x - x_max)
    return e / e.sum(axis, keepdims = True)
    raise NotImplementedError("还没实现 softmax")


def check6():
    rng = np.random.default_rng(0)

    # 6.1 普通输入：每一行的和都应该是 1
    x = rng.standard_normal((4, 5))
    s = softmax(x, axis=1)
    assert s.shape == (4, 5), f"6.1 形状不对：{s.shape}"
    assert np.allclose(s.sum(axis=1), 1.0), f"6.1 每行和应为 1，实际 {s.sum(axis=1)}"
    assert np.all(s >= 0), "6.1 概率不应为负"

    # 6.2 超大输入：不稳定的实现会在这里输出 nan
    x_big = np.array([[1000.0, 1001.0, 1002.0]])
    s_big = softmax(x_big, axis=1)
    assert np.all(np.isfinite(s_big)), f"6.2 出现 nan/inf，说明没做数值稳定处理：{s_big}"
    assert np.allclose(s_big.sum(axis=1), 1.0), "6.2 每行和应为 1"

    # 6.3 答案应与手动计算一致
    manual = np.exp(x_big - 1002.0) / np.exp(x_big - 1002.0).sum()
    assert np.allclose(s_big[0], manual), "6.3 与手算结果不一致"

    # 6.4 换个 axis 也要对
    x_col = rng.standard_normal((3, 4))
    assert np.allclose(softmax(x_col, axis=0).sum(axis=0), 1.0), "6.4 axis=0 时不对"


# ============================================================
# 运行所有检查
# ============================================================
def main():
    checks = [
        ("练习1 创建数组", check1),
        ("练习2 索引切片", check2),
        ("练习3 axis 归约", check3),
        ("练习4 广播", check4),
        ("练习5 矩阵乘法", check5),
        ("练习6 softmax", check6),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"[SKIP] {name} —— {e}")
        except AssertionError as e:
            print(f"[FAIL] {name} —— {e}")
        except Exception as e:
            print(f"[ERROR] {name} —— {type(e).__name__}: {e}")

    print(f"\n通过 {passed} / {len(checks)}")
    if passed == len(checks):
        print("全部通过，可以进入第 02 课：用 NumPy 手写线性回归 + 梯度下降")


if __name__ == "__main__":
    main()
