package Test;

import Layer.Flatten;
import Layer.MaxPool2D;
import java.util.Arrays;
import Src.Tensor;

public class Flatten_Test {
    private static int passed = 0;

    public static void main(String[] args) {
        testForwardShapeVaThuTu();
        testBatch();
        testBackward();
        testLoi();
        testGhepVoiMaxPool();
        System.out.println("\nTat ca " + passed + " test pass");
    }

    private static void check(boolean ok, String msg) {
        if (!ok) throw new AssertionError("FAIL: " + msg);
        passed++;
        System.out.println("OK: " + msg);
    }

    private static void checkData(double[] actual, double[] expected, String msg) {
        check(Arrays.equals(actual, expected),
                msg + " (actual=" + Arrays.toString(actual) + ")");
    }

    // 1) forward: shape dung, thu tu khong doi
    private static void testForwardShapeVaThuTu() {
        Tensor in = new Tensor(1, 2, 2, 2);
        for (int i = 0; i < 8; i++) in.data[i] = i + 1;

        Flatten f = new Flatten();
        Tensor out = f.forward(in);

        check(Arrays.equals(out.shape, new int[]{1, 8, 1, 1}), "forward shape = [1,8,1,1]");
        checkData(out.data, new double[]{1,2,3,4,5,6,7,8}, "forward giu nguyen thu tu c-h-w");
    }

    // 2) batch: moi anh duoi rieng, khong tron
    private static void testBatch() {
        Tensor in = new Tensor(2, 1, 2, 2);
        for (int i = 0; i < 8; i++) in.data[i] = i + 1;

        Flatten f = new Flatten();
        Tensor out = f.forward(in);

        check(Arrays.equals(out.shape, new int[]{2, 4, 1, 1}), "batch shape = [2,4,1,1]");
        checkData(out.data, new double[]{1,2,3,4,5,6,7,8}, "batch khong tron anh");
    }

    // 3) backward: la phep dao cua forward
    private static void testBackward() {
        Tensor in = new Tensor(1, 2, 2, 2);
        for (int i = 0; i < 8; i++) in.data[i] = i + 1;

        Flatten f = new Flatten();
        f.forward(in);

        Tensor dOut = new Tensor(1, 8, 1, 1);
        for (int i = 0; i < 8; i++) dOut.data[i] = (i + 1) * 10.0;

        Tensor dIn = f.backward(dOut);
        check(Arrays.equals(dIn.shape, new int[]{1, 2, 2, 2}), "backward shape = [1,2,2,2]");
        checkData(dIn.data, new double[]{10,20,30,40,50,60,70,80}, "backward dao dung thu tu");

        // goi backward lan 2 van dung (khong bi trang thai cu)
        Tensor dIn2 = f.backward(dOut);
        checkData(dIn2.data, new double[]{10,20,30,40,50,60,70,80}, "backward goi lai van dung");
    }

    // 4) cac truong hop loi
    private static void testLoi() {
        Flatten f1 = new Flatten();
        boolean threw = false;
        try {
            f1.backward(new Tensor(1, 4, 1, 1));
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "backward truoc forward phai nem loi");

        Flatten f2 = new Flatten();
        f2.forward(new Tensor(1, 2, 2, 2)); // output ky vong (1,8,1,1)
        threw = false;
        try {
            f2.backward(new Tensor(1, 8, 1, 2)); // sai chieu w
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "dOut sai shape phai nem loi");
    }

    // 5) ghep MaxPool2D -> Flatten -> backward nguoc ve
    private static void testGhepVoiMaxPool() {
        Tensor in = new Tensor(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) in.data[i] = i + 1;

        MaxPool2D pool = new MaxPool2D(2); // 2x2, stride 2
        Flatten f = new Flatten();

        Tensor pooled = pool.forward(in);
        Tensor flat = f.forward(pooled);

        check(Arrays.equals(flat.shape, new int[]{1, 4, 1, 1}), "pool->flatten shape = [1,4,1,1]");
        checkData(flat.data, new double[]{6, 8, 14, 16}, "pool->flatten giu dung 4 max");

        Tensor dOut = new Tensor(1, 4, 1, 1);
        Arrays.fill(dOut.data, 1.0);

        Tensor dFlat = f.backward(dOut);
        Tensor dIn = pool.backward(dFlat);

        check(Arrays.equals(dIn.shape, new int[]{1, 1, 4, 4}), "backward ve shape [1,1,4,4]");

        double[] expected = new double[16];
        expected[5] = 1.0;   // gia tri 6
        expected[7] = 1.0;   // gia tri 8
        expected[13] = 1.0;  // gia tri 14
        expected[15] = 1.0;  // gia tri 16
        checkData(dIn.data, expected, "gradient chi roi vao 4 o max");
    }
}
