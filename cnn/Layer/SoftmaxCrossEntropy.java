package Layer;

import Src.Tensor;

/*
 * Gộp Softmax + CrossEntropy thành một khối để ổn định số học.
 * Không implements Layer vì forward cần thêm nhãn (labels).
 *
 *   forward(logits, labels) -> loss trung bình trên batch
 *   backward()              -> dLogits = (softmax - oneHot) / n
 *
 * logits phải có shape (n, classes, 1, 1) — đúng output của Dense.
 */
public class SoftmaxCrossEntropy {
    private Tensor _probs; // softmax của lần forward gần nhất
    private int[] _labels; // nhãn của lần forward gần nhất
    private int _n;        // batch size
    private int _classes;  // số lớp

    public double forward(Tensor logits, int[] labels) {
        if (logits.h() != 1 || logits.w() != 1) {
            throw new IllegalArgumentException("logits phai co shape (n, classes, 1, 1)");
        }
        int n = logits.n();
        int classes = logits.c();
        if (labels.length != n) {
            throw new IllegalArgumentException(
                    "So nhan khong khop batch: " + labels.length + " != " + n);
        }

        Tensor probs = new Tensor(n, classes, 1, 1);
        double totalLoss = 0.0;

        for (int ni = 0; ni < n; ni++) {
            int label = labels[ni];
            if (label < 0 || label >= classes) {
                throw new IllegalArgumentException("Nhan ngoai pham vi: " + label);
            }

            // trừ max trước khi exp để tránh tràn số
            double max = Double.NEGATIVE_INFINITY;
            for (int ci = 0; ci < classes; ci++) {
                max = Math.max(max, logits.get(ni, ci, 0, 0));
            }

            double sumExp = 0.0;
            for (int ci = 0; ci < classes; ci++) {
                double e = Math.exp(logits.get(ni, ci, 0, 0) - max);
                probs.set(ni, ci, 0, 0, e);
                sumExp += e;
            }
            for (int ci = 0; ci < classes; ci++) {
                probs.set(ni, ci, 0, 0, probs.get(ni, ci, 0, 0) / sumExp);
            }

            // -log(p[label]), thêm epsilon để tránh log(0)
            totalLoss += -Math.log(probs.get(ni, label, 0, 0) + 1e-12);
        }

        _probs = probs;
        _labels = labels.clone();
        _n = n;
        _classes = classes;

        return totalLoss / n; // loss trung bình trên batch
    }

    public Tensor backward() {
        if (_probs == null) {
            throw new IllegalArgumentException("Forward phai chay truoc");
        }

        Tensor dLogits = new Tensor(_n, _classes, 1, 1);
        for (int ni = 0; ni < _n; ni++) {
            for (int ci = 0; ci < _classes; ci++) {
                double grad = _probs.get(ni, ci, 0, 0)
                        - (ci == _labels[ni] ? 1.0 : 0.0);

                // chia n để khớp loss trung bình
                dLogits.set(ni, ci, 0, 0, grad / _n);
            }
        }
        return dLogits;
    }
}