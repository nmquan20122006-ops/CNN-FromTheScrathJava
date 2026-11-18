package Layer;

import Src.Adam;
import Src.Tensor;
import java.util.Arrays;
import java.util.Random;

public class Dense implements Layer {
    private static final long serialVersionUID = 1L;

    private final int _inFeatures;   // số feature đầu vào (lấy từ Flatten)
    private final int _outFeatures;  // số lớp / số nơ-ron đầu ra

    private final double[] _weights;  // [out][in]
    private final double[] _bias;     // [out]
    private final double[] _dWeights; // gradient của _weights
    private final double[] _dBias;    // gradient của _bias

    private Tensor _input; // input (n, _inFeatures, 1, 1) của lần forward gần nhất

    // Adam lưu trạng thái riêng cho weights và bias; null nghĩa là dùng SGD
    private Adam _adamWeights;
    private Adam _adamBias;

    public Dense(int inFeatures, int outFeatures) {
        if (inFeatures <= 0) throw new IllegalArgumentException("inFeatures phai lon hon 0");
        if (outFeatures <= 0) throw new IllegalArgumentException("outFeatures phai lon hon 0");

        this._inFeatures = inFeatures;
        this._outFeatures = outFeatures;

        this._weights = new double[outFeatures * inFeatures];
        this._bias = new double[outFeatures];
        this._dWeights = new double[outFeatures * inFeatures];
        this._dBias = new double[outFeatures];

        initWeights();
    }

    // He init: std = sqrt(2 / fanIn)
    private void initWeights() {
        Random rnd = new Random(42); // seed cố định để test lặp lại được
        double std = Math.sqrt(2.0 / _inFeatures);
        for (int i = 0; i < _weights.length; i++) {
            _weights[i] = rnd.nextGaussian() * std;
        }
        // _bias mặc định đã là 0
    }

    @Override
    public Tensor forward(Tensor input) {
        // Dense chỉ nhận vector: (n, features, 1, 1) do Flatten tạo ra
        if (input.h() != 1 || input.w() != 1) {
            throw new IllegalArgumentException(
                    "Dense can input (n, features, 1, 1), nhan duoc h=" + input.h() + ", w=" + input.w());
        }
        if (input.c() != _inFeatures) {
            throw new IllegalArgumentException("So feature khong khop: " + input.c() + " != " + _inFeatures);
        }

        int n = input.n();
        Tensor out = new Tensor(n, _outFeatures, 1, 1);
        _input = input; // giữ tham chiếu để backward dùng lại

        for (int ni = 0; ni < n; ni++) {
            for (int o = 0; o < _outFeatures; o++) {
                double sum = _bias[o];
                for (int i = 0; i < _inFeatures; i++) {
                    sum += input.get(ni, i, 0, 0) * _weights[weightIndex(o, i)];
                }
                out.set(ni, o, 0, 0, sum);
            }
        }
        return out;
    }

    @Override
    public Tensor backward(Tensor dOut) {
        if (_input == null) {
            throw new IllegalArgumentException("Forward phai chay truoc");
        }
        int n = _input.n();
        if (dOut.n() != n || dOut.c() != _outFeatures || dOut.h() != 1 || dOut.w() != 1) {
            throw new IllegalArgumentException("dOut khong khop shape cua output");
        }

        zeroGrad(); // mỗi lần backward tính lại gradient từ đầu
        Tensor dInput = new Tensor(n, _inFeatures, 1, 1); // mặc định toàn 0

        for (int ni = 0; ni < n; ni++) {
            for (int o = 0; o < _outFeatures; o++) {
                double grad = dOut.get(ni, o, 0, 0);
                if (grad == 0) continue;

                _dBias[o] += grad;

                for (int i = 0; i < _inFeatures; i++) {
                    double x = _input.get(ni, i, 0, 0);

                    // dW += dOut * input
                    _dWeights[weightIndex(o, i)] += grad * x;

                    // dInput += dOut * W, cộng dồn vì nhiều output dùng chung input
                    dInput.data[ni * _inFeatures + i] += grad * _weights[weightIndex(o, i)];
                }
            }
        }
        return dInput;
    }

    @Override
    public void setOptimizer(String name) {
        if ("adam".equals(name)) {
            _adamWeights = new Adam(_weights.length);
            _adamBias = new Adam(_bias.length);
        } else if ("sgd".equals(name)) {
            _adamWeights = null;
            _adamBias = null;
        } else {
            throw new IllegalArgumentException("Optimizer khong hop le: " + name);
        }
    }

    @Override
    public void update(double learningRate) {
        if (_adamWeights != null) {
            _adamWeights.update(_weights, _dWeights, learningRate);
            _adamBias.update(_bias, _dBias, learningRate);
            return;
        }
        for (int i = 0; i < _weights.length; i++) {
            _weights[i] -= learningRate * _dWeights[i]; // chú ý: _dWeights, không phải _dBias
        }
        for (int o = 0; o < _bias.length; o++) {
            _bias[o] -= learningRate * _dBias[o];
        }
    }

    private void zeroGrad() {
        Arrays.fill(_dWeights, 0.0);
        Arrays.fill(_dBias, 0.0);
    }

    // index = o * inFeatures + i
    private int weightIndex(int o, int i) {
        return o * _inFeatures + i;
    }

    @Override
    public void clearCache() {
        _input = null;
        _adamWeights = null; // không lưu trạng thái optimizer vào file model
        _adamBias = null;
    }
}