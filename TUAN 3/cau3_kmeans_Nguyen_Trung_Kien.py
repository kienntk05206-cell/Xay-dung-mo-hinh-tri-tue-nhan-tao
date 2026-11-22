"""Câu 3: K-Means trên 1.000 điểm nguyên trong không gian 100 chiều."""
from pathlib import Path
import json
import platform

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits


def main():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=(1000, 100), dtype=np.int64)
    assert data.shape == (1000, 100)
    assert np.issubdtype(data.dtype, np.integer)
    assert data.min() >= 0 and data.max() <= 255
    lines = []

    def report(text=""):
        print(text)
        lines.append(text)

    report("CAU 3 - K-MEANS: 1000 DIEM, 100 CHIEU, TOA DO NGUYEN 0..255")
    report(f"Python {platform.python_version()}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}")
    report("Du lieu phan phoi deu, seed=42. Hai gia tri K dung cung mot bo du lieu.")
    results = {}
    # Một luồng giúp tái lập kết quả và tránh dùng quá nhiều tài nguyên.
    with threadpool_limits(limits=1):
        for k in (2, 3):
            model = KMeans(n_clusters=k, random_state=42, n_init=20,
                           max_iter=300, algorithm="lloyd", tol=1e-4)
            labels = model.fit_predict(data)
            counts = np.bincount(labels, minlength=k)
            wcss = float(model.inertia_)
            score = float(silhouette_score(data, labels, metric="euclidean"))
            # Kiểm chứng WCSS bằng tổng bình phương khoảng cách tới tâm.
            manual_wcss = float(np.square(data - model.cluster_centers_[labels]).sum())
            assert np.isclose(wcss, manual_wcss, rtol=1e-10)
            assert counts.sum() == 1000 and np.all(counts > 0)
            assert model.cluster_centers_.shape == (k, 100)
            assert -1 <= score <= 1 and np.isfinite(wcss)
            report(f"\nK = {k}")
            for i, center in enumerate(model.cluster_centers_):
                report(f"Cum {i + 1}: {counts[i]} diem")
                # In từng phần tử để bảo đảm đủ 100 tọa độ, không có dấu ...
                report(f"Tam {i + 1} (100 toa do): [" + ", ".join(f"{v:.6f}" for v in center) + "]")
            report(f"WCSS = {wcss:.6f}")
            report(f"Silhouette Score = {score:.8f}")
            results[str(k)] = dict(counts=counts.tolist(), centers=model.cluster_centers_.tolist(),
                                   wcss=wcss, silhouette=score, iterations=int(model.n_iter_))

    reduction = 100 * (results["2"]["wcss"] - results["3"]["wcss"]) / results["2"]["wcss"]
    best = max(results, key=lambda k: results[k]["silhouette"])
    report("\nSO SANH VA NHAN XET")
    report(f"WCSS giam {reduction:.4f}% khi tang K tu 2 len 3.")
    report("WCSS thuong giam khi tang K; rieng dieu nay khong chung minh K=3 tot hon.")
    report(f"K={best} co Silhouette cao hon trong hai lan chay nay.")
    report("Silhouette gan 0 cho thay cac cum chong lan, khong tach biet ro.")
    report("Du lieu duoc sinh ngau nhien deu, khong cai san cau truc cum; K-Means van chia du lieu thanh K nhom.")
    report("Trong 100 chieu, khoang cach co xu huong tap trung. Khong du co so khang dinh co 2 hay 3 cum tu nhien.")
    report("Day la ket qua tren mot seed; nhan cum co the hoan vi va ket qua co the thay doi theo seed/phien ban.")
    report("\nTEST PASS: kich thuoc, mien gia tri, 100 toa do/tam, tong 1000 diem, WCSS va Silhouette hop le.")
    folder = Path(__file__).resolve().parent
    (folder / "ket_qua.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "ket_qua.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(folder / "du_lieu.npy", data)


if __name__ == "__main__":
    main()
