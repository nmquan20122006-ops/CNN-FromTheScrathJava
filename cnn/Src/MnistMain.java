package Src;

import Layer.Conv2D;
import Layer.Dense;
import Layer.Flatten;
import Layer.MaxPool2D;
import Layer.ReLU;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import javax.imageio.ImageIO;

/*
 * Demo train MNIST bằng CNN tự code.
 * Dữ liệu mặc định nằm ở data/mnist/ (đã nén .gz).
 */
public class MnistMain {
    private static final String DIR = "data/mnist/";
    private static final String TRAIN_IMG = DIR + "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LBL = DIR + "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMG = DIR + "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LBL = DIR + "t10k-labels-idx1-ubyte.gz";

    public static boolean isAvailable() {
        return Files.exists(Paths.get(TRAIN_IMG)) && Files.exists(Paths.get(TRAIN_LBL))
            && Files.exists(Paths.get(TEST_IMG)) && Files.exists(Paths.get(TEST_LBL));
    }

    public static void main(String[] args) {
        if (!isAvailable()) {
            System.out.println("Khong thay du lieu MNIST trong " + DIR);
            return;
        }
        // Adam hội tụ nhanh hơn SGD; lr = 0.01 là giá trị thử nghiệm tốt nhất.
        // decay giúp lr giảm dần, tránh dao động/overfit ở cuối.
        run(20000, 2000, 8, 0.01, 64, "adam", 0.3);
    }

    // Giữ lại bản gọi cũ (mặc định SGD) để dễ thử nghiệm
    public static void run(int trainLimit, int testLimit, int epochs, double lr, int batchSize) {
        run(trainLimit, testLimit, epochs, lr, batchSize, "sgd");
    }

    public static void run(int trainLimit, int testLimit, int epochs, double lr, int batchSize,
                           String optimizer) {
        run(trainLimit, testLimit, epochs, lr, batchSize, optimizer, 0.0);
    }

    public static void run(int trainLimit, int testLimit, int epochs, double lr, int batchSize,
                           String optimizer, double lrDecay) {
        MnistLoader.Data train = MnistLoader.load(TRAIN_IMG, TRAIN_LBL, trainLimit);
        MnistLoader.Data test = MnistLoader.load(TEST_IMG, TEST_LBL, testLimit);

        System.out.println("MNIST: train = " + train.x.n() + " anh, test = " + test.x.n() + " anh");
        System.out.println("Ket truc: Conv(1->8) - Pool - Conv(8->16) - Pool - Flatten(784) - Dense(10)");

        Network net = new Network();
        net.addLayer(new Conv2D(1, 8, 3, 1, 1));  // (n, 8, 28, 28)
        net.addLayer(new ReLU());
        net.addLayer(new MaxPool2D(2));            // (n, 8, 14, 14)
        net.addLayer(new Conv2D(8, 16, 3, 1, 1)); // (n, 16, 14, 14)
        net.addLayer(new ReLU());
        net.addLayer(new MaxPool2D(2));            // (n, 16, 7, 7)
        net.addLayer(new Flatten());               // (n, 784, 1, 1)
        net.addLayer(new Dense(16 * 7 * 7, 10));   // (n, 10, 1, 1)

        Trainer trainer = new Trainer(net, batchSize, lr, 123L, optimizer, lrDecay);

        long startTime = System.currentTimeMillis();
        trainer.train(train.x, train.y, test.x, test.y, epochs);
        double seconds = (System.currentTimeMillis() - startTime) / 1000.0;

        System.out.printf("Thoi gian train: %.2f giay%n", seconds);
        System.out.printf("Accuracy cuoi tren test: %.2f%%%n",
                trainer.accuracy(test.x, test.y) * 100.0);

        // Lưu model để chạy dự đoán ở lần sau, không cần train lại
        Model.save(net, "model.bin");
        exportSamples(test);
    }

    // Xuất vài ảnh test thành PNG để thử lệnh predict ngay
    private static void exportSamples(MnistLoader.Data test) {
        File dir = new File("data/samples");
        if (!dir.exists()) dir.mkdirs();

        int count = Math.min(5, test.x.n());
        for (int i = 0; i < count; i++) {
            BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
            for (int r = 0; r < 28; r++) {
                for (int c = 0; c < 28; c++) {
                    int v = (int) Math.round(test.x.data[i * 28 * 28 + r * 28 + c] * 255.0);
                    v = Math.max(0, Math.min(255, v));
                    img.setRGB(c, r, (255 << 24) | (v << 16) | (v << 8) | v);
                }
            }
            try {
                ImageIO.write(img, "png", new File(dir, "sample_" + i + "_label" + test.y[i] + ".png"));
            } catch (IOException e) {
                System.out.println("Khong luu duoc anh mau: " + e.getMessage());
            }
        }
        System.out.println("Da xuat " + count + " anh mau vao data/samples/");
    }
}
