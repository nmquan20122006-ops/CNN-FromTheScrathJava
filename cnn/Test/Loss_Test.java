package Test;

import Layer.SoftmaxCrossEntropy;
import Src.Tensor;

public class Loss_Test {
    private static int passed = 0;

    public static void main(String[] args) {
        testGradientTongBang0();
        testLossDeu();
        testLossDungNhan();
        testGradientKhopCongThuc();
        testGradientCheckSoHoc();
        testBatchMean();
        testLoi();
        System.out.println("\nTat ca " + passed + " test pass");
    }

    private static void check(boolean ok, String msg) {
        if (!ok) throw new AssertionError("FAIL: " + msg);
        passed++;
        System.out.println("OK: " + msg);
    }

    private static void checkApprox(double actual, double expected, String msg) {
        check(Math.abs(actual - expected) < 1e-9,
                msg + " (actual=" + actual + ", expected=" + expected + ")");
    }

    private static double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) max = Math.max(max, v);

        double[] e = new double[z.length];
        double sum = 0.0;
        for (int i = 0; i < z.length; i++) {
            e[i] = Math.exp(z[i] - max);
            sum += e[i];
        }
        for (int i = 0; i < z.length; i++) e[i] /= sum;
        return e;
    }

    // sum(softmax - onehot) = 1 - 1 = 0
    private static void testGradientTongBang0() {
        Tensor logits = new Tensor(1, 3, 1, 1);
        logits.set(0, 0, 0, 0, 1.0);
        logits.set(0, 1, 0, 0, 2.0);
        logits.set(0, 2, 0, 0, 3.0);

        SoftmaxCrossEntropy fn = new SoftmaxCrossEntropy();
        fn.forward(logits, new int[]{1});
        Tensor d = fn.backward();

        double sum = 0.0;
        for (int i = 0; i < 3; i++) sum += d.data[i];
        check(Math.abs(sum) < 1e-12, "tong gradient theo lop = 0 (sum=" + sum + ")");
    }

    // logits đều -> loss = ln(classes)
    private static void testLossDeu() {
        SoftmaxCrossEntropy fn = new SoftmaxCrossEntropy();
        double loss = fn.forward(new Tensor(1, 3, 1, 1), new int[]{0});
        checkApprox(loss, Math.log(3.0), "logits deu -> loss = ln(3)");
    }

    // dự đoán đúng lớp phải có loss nhỏ hơn dự đoán sai
    private static void testLossDungNhan() {
        Tensor logits = new Tensor(1, 3, 1, 1);
        logits.set(0, 2, 0, 0, 6.0);

        double lossCorrect = new SoftmaxCrossEntropy().forward(logits, new int[]{2});
        double lossWrong = new SoftmaxCrossEntropy().forward(logits, new int[]{0});

        check(lossCorrect < lossWrong,
                "loss khi dung nhan nho hon (" + lossCorrect + " < " + lossWrong + ")");
    }

    // dLogits = softmax - onehot  (n = 1)
    private static void testGradientKhopCongThuc() {
        double[] z = {1.0, 2.0, 3.0};
        Tensor logits = new Tensor(1, 3, 1, 1);
        for (int i = 0; i < 3; i++) logits.set(0, i, 0, 0, z[i]);

        SoftmaxCrossEntropy fn = new SoftmaxCrossEntropy();
        fn.forward(logits, new int[]{2});
        Tensor d = fn.backward();

        double[] p = softmax(z);
        for (int i = 0; i < 3; i++) {
            double expected = p[i] - (i == 2 ? 1.0 : 0.0);
            checkApprox(d.data[i], expected, "dLogits[" + i + "] khop softmax-onehot");
        }
    }

    // sai phân số so với dLogits
    private static void testGradientCheckSoHoc() {
        Tensor logits = new Tensor(1, 3, 1, 1);
        logits.set(0, 0, 0, 0, 0.5);
        logits.set(0, 1, 0, 0, -1.0);
        logits.set(0, 2, 0, 0, 2.0);
        int[] labels = {1};

        SoftmaxCrossEntropy fn = new SoftmaxCrossEntropy();
        fn.forward(logits, labels);
        Tensor d = fn.backward();

        double eps = 1e-6;
        double maxErr = 0.0;
        for (int idx = 0; idx < 3; idx++) {
            double old = logits.data[idx];

            logits.data[idx] = old + eps;
            double plus = fn.forward(logits, labels);

            logits.data[idx] = old - eps;
            double minus = fn.forward(logits, labels);

            logits.data[idx] = old;

            double numeric = (plus - minus) / (2 * eps);
            maxErr = Math.max(maxErr, Math.abs(numeric - d.data[idx]));
        }
        check(maxErr < 1e-6, "gradient check so hoc, maxErr = " + maxErr);
    }

    // batch: loss và gradient đều lấy trung bình
    private static void testBatchMean() {
        double[] z = {1.0, 2.0, 3.0};
        Tensor logits = new Tensor(2, 3, 1, 1);
        for (int ni = 0; ni < 2; ni++) {
            for (int ci = 0; ci < 3; ci++) {
                logits.set(ni, ci, 0, 0, z[ci]);
            }
        }
        int[] labels = {0, 2};

        SoftmaxCrossEntropy fn = new SoftmaxCrossEntropy();
        double loss = fn.forward(logits, labels);

        double[] p = softmax(z);
        double expectedLoss = (-Math.log(p[0]) - Math.log(p[2])) / 2.0;
        checkApprox(loss, expectedLoss, "loss batch = trung binh 2 mau");

        Tensor d = fn.backward();
        for (int ci = 0; ci < 3; ci++) {
            checkApprox(d.get(0, ci, 0, 0),
                    (p[ci] - (ci == 0 ? 1.0 : 0.0)) / 2.0,
                    "dLogits mau 0, lop " + ci);
            checkApprox(d.get(1, ci, 0, 0),
                    (p[ci] - (ci == 2 ? 1.0 : 0.0)) / 2.0,
                    "dLogits mau 1, lop " + ci);
        }
    }

    private static void testLoi() {
        SoftmaxCrossEntropy f1 = new SoftmaxCrossEntropy();

        boolean threw = false;
        try {
            f1.forward(new Tensor(1, 3, 1, 1), new int[]{0, 1});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "so nhan khong khop batch phai loi");

        threw = false;
        try {
            f1.forward(new Tensor(1, 3, 1, 1), new int[]{5});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "nhan ngoai pham vi phai loi");

        threw = false;
        try {
            new SoftmaxCrossEntropy().backward();
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "backward truoc forward phai loi");

        threw = false;
        try {
            f1.forward(new Tensor(1, 3, 2, 1), new int[]{0});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "logits h/w khac 1 phai loi");
    }
}
