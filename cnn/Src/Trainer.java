package Src;

import Layer.SoftmaxCrossEntropy;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Random;

/*
 * Vòng train dùng chung cho mọi mạng trong project.
 *
 *   - chia dữ liệu thành mini-batch
 *   - shuffle mỗi epoch
 *   - in loss / accuracy của train và (tuỳ chọn) validation
 *
 * Giả định: mạng kết thúc bằng Dense → logits shape (n, classes, 1, 1),
 * loss dùng SoftmaxCrossEntropy (loss = trung bình trên batch).
 */
public class Trainer {
    private Network _network;
    private Network _bestNetwork;      // checkpoint có val accuracy cao nhất
    private double _bestValAcc = -1.0;
    private final SoftmaxCrossEntropy _lossFn;
    private final int _batchSize;
    private final double _learningRate;
    private final String _optimizer;
    private final double _lrDecay; // lr_epoch = lr / (1 + decay*(epoch-1))
    private final Random _rng;

    public Trainer(Network network, int batchSize, double learningRate) {
        this(network, batchSize, learningRate, 42L, "sgd", 0.0);
    }

    public Trainer(Network network, int batchSize, double learningRate, long seed) {
        this(network, batchSize, learningRate, seed, "sgd", 0.0);
    }

    public Trainer(Network network, int batchSize, double learningRate, long seed, String optimizer) {
        this(network, batchSize, learningRate, seed, optimizer, 0.0);
    }

    public Trainer(Network network, int batchSize, double learningRate, long seed, String optimizer,
                   double lrDecay) {
        if (network == null) throw new IllegalArgumentException("network khong duoc null");
        if (batchSize <= 0) throw new IllegalArgumentException("batchSize phai lon hon 0");
        if (learningRate <= 0) throw new IllegalArgumentException("learningRate phai lon hon 0");
        if (!"sgd".equals(optimizer) && !"adam".equals(optimizer)) {
            throw new IllegalArgumentException("Optimizer khong hop le: " + optimizer);
        }

        this._network = network;
        this._lossFn = new SoftmaxCrossEntropy();
        this._batchSize = batchSize;
        this._learningRate = learningRate;
        this._optimizer = optimizer;
        this._lrDecay = lrDecay;
        this._rng = new Random(seed);
        network.setOptimizer(optimizer);
    }

    public void train(Tensor trainX, int[] trainY, int epochs) {
        train(trainX, trainY, null, null, epochs);
    }

    public void train(Tensor trainX, int[] trainY, Tensor valX, int[] valY, int epochs) {
        checkShapes(trainX, trainY);
        if ((valX == null) != (valY == null)) {
            throw new IllegalArgumentException("valX va valY phai cung null hoac cung khac null");
        }
        if (valX != null) checkShapes(valX, valY);
        if (epochs <= 0) throw new IllegalArgumentException("epochs phai lon hon 0");

        int n = trainX.n();
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;

        System.out.println("Optimizer = " + _optimizer + ", lr = " + _learningRate
                + ", batch = " + _batchSize);

        // Trạng thái trước khi train
        System.out.printf("epoch   0 | train loss %.4f acc %6.2f%%",
                evaluateLoss(trainX, trainY), accuracy(trainX, trainY) * 100.0);
        if (valX != null) {
            System.out.printf(" | val loss %.4f acc %6.2f%%",
                    evaluateLoss(valX, valY), accuracy(valX, valY) * 100.0);
        }
        System.out.println();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            long startTime = System.currentTimeMillis();
            shuffle(order);

            double lossSum = 0.0;
            int correct = 0;

            for (int start = 0; start < n; start += _batchSize) {
                int end = Math.min(start + _batchSize, n);
                int[] idx = Arrays.copyOfRange(order, start, end);

                Tensor xb = sliceBatch(trainX, idx);
                int[] yb = sliceLabels(trainY, idx);

                Tensor logits = _network.forward(xb);
                double batchLoss = _lossFn.forward(logits, yb);
                Tensor dLogits = _lossFn.backward();

                _network.backward(dLogits);
                _network.update(currentLearningRate(epoch));

                lossSum += batchLoss * (end - start);
                correct += countCorrect(logits, yb);
            }

            System.out.printf("epoch %3d | train loss %.4f acc %6.2f%%",
                    epoch, lossSum / n, correct * 100.0 / n);
            if (valX != null) {
                double valLoss = evaluateLoss(valX, valY);
                double valAcc = accuracy(valX, valY);
                System.out.printf(" | val loss %.4f acc %6.2f%%", valLoss, valAcc * 100.0);

                // Lưu lại mạng tốt nhất (* = epoch này tốt hơn các epoch trước)
                if (valAcc > _bestValAcc) {
                    _bestValAcc = valAcc;
                    _bestNetwork = copyNetwork(_network);
                    System.out.print(" *");
                }
            }
            System.out.printf(" | %d ms%n", System.currentTimeMillis() - startTime);
        }

        if (_bestNetwork != null) {
            _network = _bestNetwork; // dùng checkpoint tốt nhất, không phải epoch cuối
            System.out.printf("Da chon checkpoint tot nhat: val acc %.2f%%%n", _bestValAcc * 100.0);
        }
    }

    public double evaluateLoss(Tensor x, int[] labels) {
        checkShapes(x, labels);
        int n = x.n();
        double lossSum = 0.0;

        for (int start = 0; start < n; start += _batchSize) {
            int end = Math.min(start + _batchSize, n);
            int[] idx = range(start, end);

            Tensor xb = sliceBatch(x, idx);
            int[] yb = sliceLabels(labels, idx);

            double batchLoss = _lossFn.forward(_network.forward(xb), yb);
            lossSum += batchLoss * (end - start);
        }
        return lossSum / n;
    }

    public double accuracy(Tensor x, int[] labels) {
        checkShapes(x, labels);
        int n = x.n();
        int correct = 0;

        for (int start = 0; start < n; start += _batchSize) {
            int end = Math.min(start + _batchSize, n);
            int[] idx = range(start, end);

            Tensor xb = sliceBatch(x, idx);
            int[] yb = sliceLabels(labels, idx);

            correct += countCorrect(_network.forward(xb), yb);
        }
        return (double) correct / n;
    }

    public int[] predict(Tensor x) {
        int n = x.n();
        int[] result = new int[n];

        for (int start = 0; start < n; start += _batchSize) {
            int end = Math.min(start + _batchSize, n);
            int[] idx = range(start, end);

            Tensor xb = sliceBatch(x, idx);
            Tensor logits = _network.forward(xb);

            for (int i = 0; i < idx.length; i++) {
                result[idx[i]] = argmax(logits, i);
            }
        }
        return result;
    }

    // ===== helper =====

    // Sao chép sâu mạng bằng serialize, vì Network/Tensor/layer đều Serializable
    private Network copyNetwork(Network src) {
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            new ObjectOutputStream(bos).writeObject(src);
            ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bos.toByteArray()));
            return (Network) ois.readObject();
        } catch (Exception e) {
            throw new RuntimeException("Khong the sao chep network: " + e.getMessage(), e);
        }
    }

    private double currentLearningRate(int epoch) {
        return _learningRate / (1.0 + _lrDecay * (epoch - 1));
    }

    private void checkShapes(Tensor x, int[] labels) {
        if (x == null || labels == null) {
            throw new IllegalArgumentException("x va labels khong duoc null");
        }
        if (x.n() != labels.length) {
            throw new IllegalArgumentException(
                    "So anh khong khop so nhan: " + x.n() + " != " + labels.length);
        }
    }

    // Lấy các ảnh theo index thành một batch mới (copy dữ liệu)
    private Tensor sliceBatch(Tensor x, int[] idx) {
        int perSample = x.c() * x.h() * x.w();
        Tensor batch = new Tensor(idx.length, x.c(), x.h(), x.w());

        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(x.data, idx[i] * perSample,
                             batch.data, i * perSample, perSample);
        }
        return batch;
    }

    private int[] sliceLabels(int[] labels, int[] idx) {
        int[] result = new int[idx.length];
        for (int i = 0; i < idx.length; i++) result[i] = labels[idx[i]];
        return result;
    }

    private int[] range(int start, int end) {
        int[] result = new int[end - start];
        for (int i = 0; i < result.length; i++) result[i] = start + i;
        return result;
    }

    private void shuffle(int[] a) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = _rng.nextInt(i + 1);
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    private int countCorrect(Tensor logits, int[] labels) {
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            if (argmax(logits, i) == labels[i]) correct++;
        }
        return correct;
    }

    private int argmax(Tensor logits, int sample) {
        int best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int ci = 0; ci < logits.c(); ci++) {
            double v = logits.get(sample, ci, 0, 0);
            if (v > bestVal) {
                bestVal = v;
                best = ci;
            }
        }
        return best;
    }
}
