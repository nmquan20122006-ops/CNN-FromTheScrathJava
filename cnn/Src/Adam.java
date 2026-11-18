package Src;

import java.io.Serializable;

/*
 * Adam optimizer — cách cập nhật trọng số dùng cả:
 *   - m: trung bình trượt của gradient (giống momentum)
 *   - v: trung bình trượt của gradient^2 (tự chỉnh tốc độ cho từng tham số)
 *
 * Công thức (t = số lần update):
 *   m = b1*m + (1-b1)*g
 *   v = b2*v + (1-b2)*g*g
 *   m^ = m / (1 - b1^t),  v^ = v / (1 - b2^t)   <- sửa sai lệch lúc mới đầu
 *   w = w - lr * m^ / (sqrt(v^) + eps)
 *
 * Mảng m, v có cùng kích thước với mảng trọng số/gradient.
 */
public class Adam implements Serializable {
    private static final long serialVersionUID = 1L;

    private static final double BETA1 = 0.9;
    private static final double BETA2 = 0.999;
    private static final double EPS = 1e-8;

    private final double[] _m;
    private final double[] _v;
    private int _t = 0; // số lần update

    public Adam(int size) {
        if (size <= 0) throw new IllegalArgumentException("size phai lon hon 0");
        this._m = new double[size];
        this._v = new double[size];
    }

    public void update(double[] params, double[] grads, double learningRate) {
        if (params.length != grads.length || params.length != _m.length) {
            throw new IllegalArgumentException("params, grads va state phai cung kich thuoc");
        }

        _t++;
        double corr1 = 1.0 - Math.pow(BETA1, _t);
        double corr2 = 1.0 - Math.pow(BETA2, _t);

        for (int i = 0; i < params.length; i++) {
            double g = grads[i];
            _m[i] = BETA1 * _m[i] + (1.0 - BETA1) * g;
            _v[i] = BETA2 * _v[i] + (1.0 - BETA2) * g * g;

            double mHat = _m[i] / corr1;
            double vHat = _v[i] / corr2;
            params[i] -= learningRate * mHat / (Math.sqrt(vHat) + EPS);
        }
    }
}
