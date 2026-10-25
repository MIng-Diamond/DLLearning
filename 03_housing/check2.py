"""
项目一 · 第 2 节 —— 检查
==========================
    cd ~/projects/learn/DeepLearning/03_housing
    python check2.py
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score as sk_r2

from data import load_data, split_and_standardize
from model import predict, train
from evaluate import rmse, r2_score
from predict import predict_raw


def check_metrics():
    """你的 rmse / r2 必须和 sklearn 的工业标准实现一致"""
    rng = np.random.default_rng(0)

    for t in range(3):
        yt = rng.normal(2.0, 1.0, 300)
        yp = yt + rng.normal(0, 0.5, 300)

        got = rmse(yp, yt)
        want = np.sqrt(mean_squared_error(yt, yp))
        assert abs(got - want) < 1e-10, \
            f"rmse 不对：你算 {got:.10f}，sklearn 算 {want:.10f}"

        got = r2_score(yp, yt)
        want = sk_r2(yt, yp)
        assert abs(got - want) < 1e-10, \
            f"r2_score 不对：你算 {got:.10f}，sklearn 算 {want:.10f}"

    # —— 概念检查：预测「均值」时 R² 必须 ≈ 0 ——
    yt = rng.normal(5.0, 2.0, 500)
    yp_mean = np.full_like(yt, yt.mean())
    r = r2_score(yp_mean, yt)
    assert abs(r) < 1e-10, \
        f"永远预测均值的「傻瓜模型」R² 应该是 0，你算出来是 {r:.6f} —— 公式的分母写对了吗？"

    # —— 概念检查：完美预测时 R² = 1 ——
    assert abs(r2_score(yt, yt) - 1.0) < 1e-10, "完美预测时 R² 应为 1"

    # —— RMSE 的单位是目标变量的单位 ——
    assert abs(rmse(yt + 0.5, yt) - 0.5) < 1e-10, "整体偏移 0.5 时 RMSE 应为 0.5"

    print(f"    rmse / r2_score 与 sklearn 完全一致  ✅")
    print(f"    傻瓜模型（永远预测均值）R² = {r2_score(yp_mean, yt):.2e}  ✅")


def check_train():
    df = load_data()
    X_tr, y_tr, X_te, y_te, mean, std = split_and_standardize(df)

    w, b, losses = train(X_tr, y_tr, lr=0.1, epochs=2000)

    assert len(losses) == 2000, f"losses 应有 2000 个，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降"
    assert np.all(np.isfinite(losses)), "损失出现了 nan/inf —— 学习率太大了？"

    # 这里用 sklearn 的标准实现来评测，就不用依赖你写的 r2_score ——
    # 这样 model.py 可以独立验证（r2_score 由 check_metrics 单独负责）
    yp_te = predict(X_te, w, b)
    r2_tr = sk_r2(y_tr, predict(X_tr, w, b))
    r2_te = sk_r2(y_te, yp_te)
    rmse_te = float(np.sqrt(mean_squared_error(y_te, yp_te)))

    print(f"    R²(训练集) = {r2_tr:.4f}    R²(测试集) = {r2_te:.4f}")
    print(f"    RMSE(测试集) = {rmse_te:.4f} ≈ {rmse_te*10:.1f} 千美元")

    assert r2_te >= 0.60, \
        f"测试集 R² = {r2_te:.4f}，应 ≥ 0.60。检查：学习率 / 轮数 / 梯度公式的符号"
    assert r2_te < 0.95, \
        f"测试集 R² = {r2_te:.4f} 高得不正常 —— 大概率是数据泄漏（测试集信息漏进训练了）"

    return w, b, mean, std, X_te, y_te


def check_predict_raw(w, b, mean, std, X_te):
    """predict_raw 必须和「先手动标准化再 predict」得到相同结果"""
    # 取一个测试样本，反标准化还原成原始量纲
    raw = X_te[0] * std + mean
    got = predict_raw(w, b, mean, std, raw)
    want = predict(X_te[0], w, b)
    assert np.allclose(got, want, atol=1e-8), \
        f"predict_raw 结果 {got} 与手算 {want} 不一致 —— 标准化那一步对吗？"

    # 批量也要支持
    raws = X_te[:5] * std + mean
    got5 = predict_raw(w, b, mean, std, raws)
    assert np.shape(got5) == (5,), f"批量输入应返回 (5,)，实际 {np.shape(got5)}"

    print(f"    单样本: 原始特征 → 预测 {float(np.asarray(got)):.4f} (10万美元)")
    print(f"    批量 5 个样本也能处理  ✅")


def _run(name, fn):
    """跑一个检查，统一处理四种结果。返回 (是否通过, 函数返回值)。"""
    try:
        out = fn()
        print(f"[PASS] {name}")
        return True, out
    except NotImplementedError as e:
        print(f"[SKIP] {name} —— {e}")
    except AssertionError as e:
        print(f"[FAIL] {name} —— {e}")
    except Exception as e:
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行: {type(e).__name__}: {e}")
    return False, None


def main():
    passed = 0

    ok, _ = _run("⑤⑥ rmse / r2_score", check_metrics)
    if ok:
        passed += 1

    print()
    ok, got = _run("①~④ train 全流程", check_train)
    if ok:
        passed += 1

    if ok:
        w, b, mean, std, X_te, y_te = got
        print()
        ok2, _ = _run("⑦ predict_raw",
                      lambda: check_predict_raw(w, b, mean, std, X_te))
        if ok2:
            passed += 1

    print(f"\n通过 {passed} / 3")
    if passed == 3:
        print("\n第 2 节完成 ✅  可以跑 run.py 出图了")


if __name__ == "__main__":
    main()
