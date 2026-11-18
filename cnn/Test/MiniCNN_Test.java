package Test;

import Layer.Conv2D;
import Layer.Dense;
import Layer.Flatten;
import Layer.MaxPool2D;
import Layer.ReLU;
import Layer.SoftmaxCrossEntropy;
import Src.Network;
import Src.Tensor;

/*
 * Test tích hợp: ghép cả mạng nhỏ và kiểm tra loss giảm sau khi train.
 * Bài toán: 8 ảnh 4x4, 2 lớp.
 *   - lớp 0: pixel sáng thấp
 *   - lớp 1: pixel sáng cao
 */
public class MiniCNN_Test {
    public static void main(String[] args) {
        int n = 8;
        int channels = 1;
        int h = 4, w = 4;

        Tensor x = new Tensor(n, channels, h, w);
        int[] labels = new int[n];

        for (int ni = 0; ni < n; ni++) {
            labels[ni] = ni < n / 2 ? 0 : 1;
            double base = labels[ni] == 0 ? 0.0 : 1.0;
            for (int i = 0; i < h * w; i++) {
                double jitter = ((i * 7 + ni * 3) % 5) * 0.01;
                x.data[ni * h * w + i] = base + jitter;
            }
        }

        Network net = new Network();
        net.addLayer(new Conv2D(1, 2, 3, 1, 1)); // (8,2,4,4)
        net.addLayer(new ReLU());
        net.addLayer(new MaxPool2D(2));          // (8,2,2,2)
        net.addLayer(new Flatten());             // (8,8,1,1)
        net.addLayer(new Dense(8, 2));           // (8,2,1,1)

        SoftmaxCrossEntropy lossFn = new SoftmaxCrossEntropy();

        double initialLoss = lossFn.forward(net.forward(x), labels);

        double lr = 0.05;
        int epochs = 300;
        double loss = initialLoss;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Tensor logits = net.forward(x);
            loss = lossFn.forward(logits, labels);
            Tensor dLogits = lossFn.backward();
            net.backward(dLogits);
            net.update(lr);
        }

        Tensor logits = net.forward(x);
        double finalLoss = lossFn.forward(logits, labels);

        int correct = 0;
        for (int ni = 0; ni < n; ni++) {
            int predicted = argmax(logits, ni);
            if (predicted == labels[ni]) correct++;
        }

        System.out.println("initial loss = " + initialLoss);
        System.out.println("final loss   = " + finalLoss);
        System.out.println("accuracy     = " + correct + "/" + n);

        if (!(finalLoss < initialLoss * 0.5)) {
            throw new AssertionError("FAIL: loss khong giam du ("
                    + initialLoss + " -> " + finalLoss + ")");
        }
        if (correct < 7) {
            throw new AssertionError("FAIL: accuracy qua thap (" + correct + "/" + n + ")");
        }

        System.out.println("OK: mang train duoc, loss giam va du doan dung");
    }

    private static int argmax(Tensor logits, int ni) {
        int best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int ci = 0; ci < logits.c(); ci++) {
            double v = logits.get(ni, ci, 0, 0);
            if (v > bestVal) {
                bestVal = v;
                best = ci;
            }
        }
        return best;
    }
}
