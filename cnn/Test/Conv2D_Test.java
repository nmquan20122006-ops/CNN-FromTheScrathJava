package Test;

import Layer.Conv2D;
import Src.Tensor;
import java.util.Arrays;

public class Conv2D_Test {
    private static int passed = 0;

    public static void main(String[] args) {
        testForwardShape();
        testForwardPaddingBien();
        testForwardKhopTinhTay();
        testGradientCheckDInput();
        testDWvaDBias();
        testUpdateGiamLoss();
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

    private static double dot(Tensor a, Tensor b) {
        double s = 0;
        for (int i = 0; i < a.data.length; i++) s += a.data[i] * b.data[i];
        return s;
    }

    private static Tensor makeInput(int n, int c, int h, int w) {
        Tensor t = new Tensor(n, c, h, w);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = (i % 7) - 3; // trộn số âm/dương
        }
        return t;
    }

    private static void testForwardShape() {
        Conv2D c1 = new Conv2D(3, 4, 3, 1, 1); // "same" padding
        Tensor o1 = c1.forward(makeInput(2, 3, 8, 8));
        check(Arrays.equals(o1.shape, new int[]{2, 4, 8, 8}), "shape voi stride 1, padding 1 = [2,4,8,8]");

        Conv2D c2 = new Conv2D(1, 1, 3, 2, 0); // stride 2
        Tensor o2 = c2.forward(makeInput(1, 1, 8, 8));
        check(Arrays.equals(o2.shape, new int[]{1, 1, 3, 3}), "shape voi stride 2 = [1,1,3,3]");
    }

    // padding > 0 từng làm lỗi biên (ih == h) ở bản cũ
    private static void testForwardPaddingBien() {
        Conv2D c = new Conv2D(1, 1, 3, 1, 1);
        Tensor out = c.forward(makeInput(1, 1, 4, 4));
        check(Arrays.equals(out.shape, new int[]{1, 1, 4, 4}), "padding 1 khong loi bien, shape [1,1,4,4]");
    }

    // so sánh forward với phép tính tay dùng weights/bias đọc từ layer
    private static void testForwardKhopTinhTay() {
        Conv2D c = new Conv2D(2, 3, 3, 1, 1);
        Tensor in = makeInput(1, 2, 4, 4);
        Tensor out = c.forward(in);

        double[] W = c.weights();
        double[] B = c.bias();

        double maxErr = 0.0;
        for (int oc = 0; oc < 3; oc++) {
            for (int oh = 0; oh < out.h(); oh++) {
                for (int ow = 0; ow < out.w(); ow++) {
                    double sum = B[oc];
                    for (int ic = 0; ic < 2; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = oh - 1 + kh; // stride 1, padding 1
                                int iw = ow - 1 + kw;
                                if (ih < 0 || ih >= in.h() || iw < 0 || iw >= in.w()) continue;
                                int wi = ((oc * 2 + ic) * 3 + kh) * 3 + kw;
                                sum += in.get(0, ic, ih, iw) * W[wi];
                            }
                        }
                    }
                    maxErr = Math.max(maxErr, Math.abs(sum - out.get(0, oc, oh, ow)));
                }
            }
        }
        check(maxErr < 1e-12, "forward khop tinh tay, maxErr = " + maxErr);
    }

    // loss = sum(out * dOut) => dloss/dout = dOut
    private static void testGradientCheckDInput() {
        Conv2D c = new Conv2D(2, 3, 3, 1, 1);
        Tensor in = makeInput(1, 2, 4, 4);

        Tensor out = c.forward(in);
        Tensor dOut = new Tensor(out.n(), out.c(), out.h(), out.w());
        for (int i = 0; i < dOut.data.length; i++) dOut.data[i] = 1.0 + (i % 5);

        Tensor dIn = c.backward(dOut);

        double eps = 1e-6;
        double maxErr = 0.0;
        for (int idx = 0; idx < in.data.length; idx++) {
            double old = in.data[idx];

            in.data[idx] = old + eps;
            double plus = dot(c.forward(in), dOut);

            in.data[idx] = old - eps;
            double minus = dot(c.forward(in), dOut);

            in.data[idx] = old;

            double numeric = (plus - minus) / (2 * eps);
            maxErr = Math.max(maxErr, Math.abs(numeric - dIn.data[idx]));
        }
        check(maxErr < 1e-6, "gradient check dInput, maxErr = " + maxErr);
    }

    // dW = tong dOut * input, dB = tong dOut
    private static void testDWvaDBias() {
        Conv2D c = new Conv2D(1, 1, 3, 1, 1);
        Tensor in = new Tensor(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) in.data[i] = i + 1;

        Tensor out = c.forward(in);
        Tensor dOut = new Tensor(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) dOut.data[i] = (i % 5) - 2;

        c.backward(dOut);

        double[] dW = c.dWeights();
        double[] dB = c.dBias();
        double[] expectedW = new double[9];
        double expectedB = 0.0;

        for (int oh = 0; oh < 4; oh++) {
            for (int ow = 0; ow < 4; ow++) {
                double g = dOut.get(0, 0, oh, ow);
                expectedB += g;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh - 1 + kh;
                        int iw = ow - 1 + kw;
                        if (ih < 0 || ih >= 4 || iw < 0 || iw >= 4) continue;
                        expectedW[kh * 3 + kw] += g * in.get(0, 0, ih, iw);
                    }
                }
            }
        }

        double maxErrW = 0.0;
        for (int i = 0; i < 9; i++) {
            maxErrW = Math.max(maxErrW, Math.abs(expectedW[i] - dW[i]));
        }
        check(maxErrW < 1e-12, "dWeights khop tinh tay, maxErr = " + maxErrW);
        checkApprox(dB[0], expectedB, "dBias = tong dOut");
    }

    // update phải dùng _dWeights và làm giảm loss; bản cũ dùng _dBias -> AIOOBE
    private static void testUpdateGiamLoss() {
        Conv2D c = new Conv2D(1, 1, 3, 1, 1);
        Tensor in = makeInput(1, 1, 4, 4);
        Tensor dOut = new Tensor(1, 1, 4, 4);
        Arrays.fill(dOut.data, 1.0);

        double before = dot(c.forward(in), dOut);
        c.backward(dOut);
        c.update(0.01);
        double after = dot(c.forward(in), dOut);

        check(after < before, "update lam giam loss (" + before + " -> " + after + ")");
    }

    private static void testLoi() {
        boolean threw = false;
        try {
            new Conv2D(0, 1, 3);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "inChannels <= 0 phai loi");

        threw = false;
        try {
            new Conv2D(1, 1, 3, 1, -1);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "padding < 0 phai loi");

        threw = false;
        try {
            new Conv2D(2, 1, 3).forward(new Tensor(1, 1, 4, 4));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "sai so kenh input phai loi");

        threw = false;
        try {
            new Conv2D(1, 1, 5).forward(new Tensor(1, 1, 3, 3));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "input qua nho phai loi");

        Conv2D c = new Conv2D(1, 1, 3);
        threw = false;
        try {
            c.backward(new Tensor(1, 1, 2, 2));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "backward truoc forward phai loi");

        c.forward(new Tensor(1, 1, 4, 4));
        threw = false;
        try {
            c.backward(new Tensor(1, 1, 3, 3));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "dOut sai shape phai loi");
    }
}
