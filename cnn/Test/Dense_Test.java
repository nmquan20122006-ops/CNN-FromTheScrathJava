package Test;

import Layer.Dense;
import Layer.Flatten;
import Src.Tensor;
import java.util.Arrays;

public class Dense_Test {
    private static int passed = 0;

    public static void main(String[] args) {
        testForwardShape();
        testLoi();
        testGradientCheckDInput();
        testUpdateGiamLoss();
        testBiasCapNhat();
        testBackwardCongDonTheoBatch();
        testGhepFlattenDense();
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

    private static void testForwardShape() {
        Dense d = new Dense(4, 3);
        Tensor out = d.forward(new Tensor(2, 4, 1, 1));
        check(Arrays.equals(out.shape, new int[]{2, 3, 1, 1}), "forward shape = [2,3,1,1]");
    }

    private static void testLoi() {
        Dense d1 = new Dense(3, 2);
        boolean threw = false;
        try {
            d1.forward(new Tensor(1, 4, 1, 1));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "sai so feature phai nem loi");

        threw = false;
        try {
            d1.forward(new Tensor(1, 3, 2, 1));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "h/w khac 1 phai nem loi");

        Dense d2 = new Dense(3, 2);
        threw = false;
        try {
            d2.backward(new Tensor(1, 2, 1, 1));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "backward truoc forward phai nem loi");

        d2.forward(new Tensor(1, 3, 1, 1));
        threw = false;
        try {
            d2.backward(new Tensor(1, 2, 1, 2));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "dOut sai shape phai nem loi");
    }

    // loss = sum(out * dOut) => dloss/dout = dOut
    private static void testGradientCheckDInput() {
        Tensor in = new Tensor(1, 3, 1, 1);
        in.data[0] = 0.5;
        in.data[1] = -1.0;
        in.data[2] = 2.0;

        Dense d = new Dense(3, 2);
        d.forward(in);

        Tensor dOut = new Tensor(1, 2, 1, 1);
        dOut.data[0] = 1.0;
        dOut.data[1] = -2.0;

        Tensor dIn = d.backward(dOut);

        double eps = 1e-6;
        double maxErr = 0.0;
        for (int idx = 0; idx < in.data.length; idx++) {
            double old = in.data[idx];

            in.data[idx] = old + eps;
            double plus = dot(d.forward(in), dOut);

            in.data[idx] = old - eps;
            double minus = dot(d.forward(in), dOut);

            in.data[idx] = old;

            double numeric = (plus - minus) / (2 * eps);
            maxErr = Math.max(maxErr, Math.abs(numeric - dIn.data[idx]));
        }
        check(maxErr < 1e-6, "gradient check dInput, maxErr = " + maxErr);
    }

    private static void testUpdateGiamLoss() {
        Tensor in = new Tensor(1, 3, 1, 1);
        in.data[0] = 1.0;
        in.data[1] = -2.0;
        in.data[2] = 0.5;

        Dense d = new Dense(3, 2);
        Tensor dOut = new Tensor(1, 2, 1, 1);
        dOut.data[0] = 1.0;
        dOut.data[1] = -1.0;

        double before = dot(d.forward(in), dOut);
        d.backward(dOut);
        d.update(0.01);
        double after = dot(d.forward(in), dOut);

        check(after < before, "update lam giam loss (" + before + " -> " + after + ")");
    }

    // input toàn 0 => chỉ bias đóng góp; update() phải dùng đúng _dBias
    private static void testBiasCapNhat() {
        Tensor zeros = new Tensor(1, 3, 1, 1);
        Dense d = new Dense(3, 2);

        Tensor dOut = new Tensor(1, 2, 1, 1);
        dOut.data[0] = 2.0;
        dOut.data[1] = -1.0;

        d.forward(zeros);
        d.backward(dOut);
        d.update(0.5);

        Tensor out = d.forward(zeros);
        checkApprox(out.data[0], -1.0, "bias[0] = -lr * dBias[0]");
        checkApprox(out.data[1], 0.5, "bias[1] = -lr * dBias[1]");
    }

    // dBias phải cộng dồn qua batch
    private static void testBackwardCongDonTheoBatch() {
        Tensor zeros = new Tensor(2, 3, 1, 1);
        Dense d = new Dense(3, 2);

        Tensor dOut = new Tensor(2, 2, 1, 1);
        dOut.set(0, 0, 0, 0, 1.0); // mẫu 0 chỉ đóng góp cho output 0
        dOut.set(1, 1, 0, 0, 1.0); // mẫu 1 chỉ đóng góp cho output 1

        d.forward(zeros);
        d.backward(dOut);
        d.update(1.0);

        Tensor out = d.forward(zeros);
        checkApprox(out.get(0, 0, 0, 0), -1.0, "batch: output0 mau0");
        checkApprox(out.get(0, 1, 0, 0), -1.0, "batch: output1 mau0");
        checkApprox(out.get(1, 0, 0, 0), -1.0, "batch: output0 mau1");
        checkApprox(out.get(1, 1, 0, 0), -1.0, "batch: output1 mau1");
    }

    private static void testGhepFlattenDense() {
        Tensor in = new Tensor(1, 1, 2, 2);
        for (int i = 0; i < 4; i++) in.data[i] = i + 1;

        Flatten f = new Flatten();
        Dense d = new Dense(4, 2);

        Tensor flat = f.forward(in);       // (1,4,1,1)
        Tensor out = d.forward(flat);      // (1,2,1,1)
        check(Arrays.equals(out.shape, new int[]{1, 2, 1, 1}), "flatten->dense shape = [1,2,1,1]");

        Tensor dOut = new Tensor(1, 2, 1, 1);
        Arrays.fill(dOut.data, 1.0);

        Tensor dFlat = d.backward(dOut);
        Tensor dIn = f.backward(dFlat);
        check(Arrays.equals(dIn.shape, new int[]{1, 1, 2, 2}), "dense->flatten backward ve [1,1,2,2]");
    }
}
