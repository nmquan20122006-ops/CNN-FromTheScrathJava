package Src;

import Layer.Conv2D;
import Layer.Dense;
import Layer.Flatten;
import Layer.MaxPool2D;
import Layer.ReLU;
import java.util.Random;

/*
 * Demo train thật bằng mạng CNN tự code.
 *
 * Bài toán: phân loại ảnh 8x8 thành 2 lớp
 *   - lớp 0: có một đường thẳng đứng
 *   - lớp 1: có một đường thẳng ngang
 *
 * Ảnh được tạo ngẫu nhiên (có nhiễu và đổi vị trí đường), nên mạng phải học
 * đặc trưng chứ không phải nhớ từng mẫu cụ thể.
 */
public class Main {
    private static class Data {
        final Tensor x;
        final int[] y;

        Data(Tensor x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }

    public static void main(String[] args) {
        if (MnistMain.isAvailable()) {
            System.out.println("Tim thay du lieu MNIST -> train MNIST.");
            MnistMain.main(args);
            return;
        }
        System.out.println("Khong thay MNIST trong data/mnist/ -> chay demo duong thang.");
        runBarDemo();
    }

    private static void runBarDemo() {
        Data train = makeBars(200, 7L);
        Data val = makeBars(80, 99L);

        Network net = new Network();
        net.addLayer(new Conv2D(1, 4, 3, 1, 1)); // (n, 4, 8, 8)
        net.addLayer(new ReLU());
        net.addLayer(new MaxPool2D(2));         // (n, 4, 4, 4)
        net.addLayer(new Conv2D(4, 8, 3, 1, 1)); // (n, 8, 4, 4)
        net.addLayer(new ReLU());
        net.addLayer(new MaxPool2D(2));         // (n, 8, 2, 2)
        net.addLayer(new Flatten());            // (n, 32, 1, 1)
        net.addLayer(new Dense(32, 2));         // (n, 2, 1, 1)

        Trainer trainer = new Trainer(net, 16, 0.05, 123L);
        trainer.train(train.x, train.y, val.x, val.y, 25);

        int[] pred = trainer.predict(val.x);
        System.out.print("\nDu doan 12 mau dau cua validation: ");
        for (int i = 0; i < 12; i++) {
            System.out.print(pred[i] + (pred[i] == val.y[i] ? " " : "!"));
        }
        System.out.println(" (! = sai)");
        System.out.printf("Nhan that:                          ");
        for (int i = 0; i < 12; i++) System.out.print(val.y[i] + " ");
        System.out.println();
        System.out.printf("Accuracy cuoi tren validation: %.2f%%%n",
                trainer.accuracy(val.x, val.y) * 100.0);
    }

    private static Data makeBars(int count, long seed) {
        int h = 8;
        int w = 8;
        int pixelsPerImage = h * w;

        Tensor x = new Tensor(count, 1, h, w);
        int[] y = new int[count];
        Random rnd = new Random(seed);

        for (int ni = 0; ni < count; ni++) {
            int label = rnd.nextInt(2); // 0: dung, 1: ngang
            y[ni] = label;

            int offset = ni * pixelsPerImage;

            // nhiền nền nhẹ
            for (int p = 0; p < pixelsPerImage; p++) {
                x.data[offset + p] = rnd.nextGaussian() * 0.1;
            }

            if (label == 0) {
                // đường thẳng đứng ở một cột ngẫu nhiên
                int col = 1 + rnd.nextInt(w - 2);
                for (int r = 0; r < h; r++) {
                    x.data[offset + r * w + col] = 1.0 + rnd.nextGaussian() * 0.05;
                }
            } else {
                // đường thẳng ngang ở một hàng ngẫu nhiên
                int row = 1 + rnd.nextInt(h - 2);
                for (int c = 0; c < w; c++) {
                    x.data[offset + row * w + c] = 1.0 + rnd.nextGaussian() * 0.05;
                }
            }
        }
        return new Data(x, y);
    }
}
