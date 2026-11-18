package Src;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/*
 * Dự đoán một tấm ảnh bằng model đã lưu.
 *
 * Cách dùng:
 *   java -cp out Src.PredictMain model.bin image.png
 *   java -cp out Src.PredictMain model.bin --idx 123
 *
 * Ảnh đầu vào được đổi thành grayscale 28x28 rồi chia 255.
 * Nếu nền sáng (chữ đen trên nền trắng) thì tự đảo lại thành chữ trắng trên
 * nền đen, đúng với cách MNIST được huấn luyện.
 * Ảnh nên có chữ tương đối căn giữa, giống ảnh MNIST, để dự đoán tốt nhất.
 */
public class PredictMain {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Cach dung:");
            System.out.println("  java -cp out Src.PredictMain model.bin image.png");
            System.out.println("  java -cp out Src.PredictMain model.bin --idx 123");
            return;
        }

        String modelPath = args[0];
        String imageArg = args[1];
        Network net = Model.load(modelPath);
        Tensor x;

        if ("--idx".equals(imageArg)) {
            if (args.length < 3) {
                System.out.println("Thieu index anh. Vd: --idx 123");
                return;
            }
            int idx = Integer.parseInt(args[2]);
            x = mnistImage(idx);
        } else {
            try {
                x = readImage(imageArg);
            } catch (Exception e) {
                System.out.println("Doc anh that bai: " + e.getMessage());
                return;
            }
        }

        double[] probs = probabilities(net, x);
        int predicted = argmax(probs);

        System.out.println("==> Du doan: " + predicted);
        System.out.println("Xac suat:");
        for (int i = 0; i < probs.length; i++) {
            System.out.printf("  %d: %6.2f%%%n", i, probs[i] * 100.0);
        }
    }

    // Dự đoán cho code khác gọi trực tiếp (ảnh đã là tensor 1x1x28x28)
    public static int predict(Network net, Tensor x) {
        return argmax(probabilities(net, x));
    }

    public static double[] probabilities(Network net, Tensor x) {
        Tensor logits = net.forward(x);
        return softmax(logits);
    }

    private static Tensor mnistImage(int index) {
        MnistLoader.Data data = MnistLoader.load(
                "data/mnist/t10k-images-idx3-ubyte.gz",
                "data/mnist/t10k-labels-idx1-ubyte.gz",
                index + 1);

        Tensor x = new Tensor(1, 1, 28, 28);
        System.arraycopy(data.x.data, index * 28 * 28, x.data, 0, 28 * 28);
        System.out.println("Nhan that cua anh test thu " + index + ": " + data.y[index]);
        return x;
    }

    private static Tensor readImage(String path) throws Exception {
        BufferedImage src = ImageIO.read(new File(path));
        if (src == null) throw new IllegalArgumentException("Khong doc duoc anh: " + path);

        // resize + grayscale về 28x28
        BufferedImage gray = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                           RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, 28, 28, null);
        g.dispose();

        double[] raw = new double[28 * 28];
        double mean = 0.0;
        for (int i = 0; i < raw.length; i++) {
            raw[i] = (gray.getRGB(i % 28, i / 28) & 0xFF) / 255.0;
            mean += raw[i];
        }
        mean /= raw.length;

        // MNIST: chữ trắng trên nền đen. Ảnh chữ đen/nền trắng thì tự đảo lại.
        boolean invert = mean > 0.5;
        Tensor x = new Tensor(1, 1, 28, 28);
        for (int i = 0; i < raw.length; i++) {
            x.data[i] = invert ? 1.0 - raw[i] : raw[i];
        }
        return x;
    }

    private static double[] softmax(Tensor logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.data.length; i++) {
            max = Math.max(max, logits.data[i]);
        }

        double[] out = new double[logits.data.length];
        double sum = 0.0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp(logits.data[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }

    private static int argmax(double[] values) {
        int best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > bestVal) {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }
}
